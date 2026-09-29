# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import orjson
import pytest
from omegaconf import OmegaConf
from pydantic import ConfigDict

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _configured_model_name,
    _create_usersim_generator,
    _is_retryable_dependency_error,
)
from nemo_gym.base_environment_server import BaseEnvironmentServer
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from resources_servers.usersim.episode_contracts import (
    ProbeRuntimeDescriptor,
    UserSimEpisodeRequest,
    UserSimSeedResponse,
    UserSimTaskInput,
)


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _Response:
    ok = True

    def __init__(self, body: dict[str, Any], *, cookie: str | None = None) -> None:
        self.body = orjson.dumps(body)
        self.cookies = {"session": _Cookie(cookie)} if cookie is not None else {}

    async def read(self) -> bytes:
        return self.body


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict[str, Any]]]
    responses: list[_Response]

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        response = self.responses.pop(0)
        if url_path == "/seed_session":
            payload = orjson.loads(response.body)
            payload["resources_session_id"] = kwargs["json"].resources_session_id
            response.body = orjson.dumps(payload)
        elif url_path in {"/v1/agent_sessions", "/v1/agent_sessions/close"}:
            payload = orjson.loads(response.body)
            body = kwargs["json"]
            payload["agent_session_id"] = body["agent_session_id"] if isinstance(body, dict) else body.agent_session_id
            response.body = orjson.dumps(payload)
        elif url_path == "/close_session":
            payload = orjson.loads(response.body)
            payload["resources_session_id"] = kwargs["json"].resources_session_id
            response.body = orjson.dumps(payload)
        return response

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _model_response(response_id: str, text: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": f"{response_id}-message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _tool_model_response() -> dict[str, Any]:
    response = _model_response("assistant-response", "Done.")
    response["output"] = [
        {
            "id": "fc-1",
            "call_id": "call-1",
            "name": "safe_action",
            "arguments": '{"value":"x"}',
            "type": "function_call",
            "status": "completed",
        },
        {
            "call_id": "call-1",
            "output": '{"ok":true}',
            "type": "function_call_output",
        },
        *response["output"],
    ]
    return response


def _runtime_descriptor() -> ProbeRuntimeDescriptor:
    return ProbeRuntimeDescriptor.model_validate(
        {
            "probe_type": "safety_agentic",
            "assistant_tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "safe_action",
                        "description": "Perform an action.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "allowed_tool_names": ["safe_action"],
            "initial_user_message": "Perform the action.",
            "loop_policy": {
                "tool_round_mode": "multi",
                "max_assistant_activations": 3,
                "final_synthesis_without_tools": False,
                "single_user_turn": True,
                "assistant_error_behavior": "fail_episode",
                "tool_error_behavior": "return_error_payload",
                "max_tool_response_attempts": 1,
                "assistant_resampling": False,
            },
            "user_system_prompt": "",
            "assistant_system_prompt": "",
            "turn0_user_query_instruction": None,
            "user_interaction_style": "direct",
            "patience": 0.5,
            "user_turn_policy": {
                "context_compression": True,
                "wrap_up": True,
                "followup_anchor": None,
                "allowed_phrases": [],
                "script_check_ignores": [],
                "check_opening": "none",
            },
        }
    )


def _environment_server(*, token_capture: bool = False) -> tuple[UserSimEnvironmentServer, _Client]:
    servers = {
        "resources": {"resources_servers": {"usersim": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}},
        "user": {
            "responses_api_agents": {
                "user": {
                    "host": "user",
                    "port": 8001,
                    "entrypoint": "app.py",
                    "token_id_capture": token_capture,
                    "model_server": {"type": "responses_api_models", "name": "policy"},
                }
            }
        },
        "assistant": {
            "responses_api_agents": {
                "assistant": {
                    "host": "assistant",
                    "port": 8002,
                    "entrypoint": "app.py",
                    "token_id_capture": token_capture,
                    "model_server": {"type": "responses_api_models", "name": "policy"},
                }
            }
        },
        "policy": {
            "responses_api_models": {
                "vllm_model": {
                    "host": "policy",
                    "port": 8004,
                    "entrypoint": "app.py",
                    "model": "nvidia/example-assistant-model",
                }
            }
        },
        "support": {
            "responses_api_models": {
                "vllm_model": {
                    "host": "support",
                    "port": 8003,
                    "entrypoint": "app.py",
                    "model": "nvidia/example-support-model",
                }
            }
        },
        "token_id_capture": {"enabled": token_capture},
    }
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(servers),
        calls=[],
        responses=[],
    )
    config = UserSimEnvironmentServerConfig(
        name="usersim-environment",
        host="environment",
        port=8005,
        entrypoint="app.py",
        cleanup_timeout_seconds=10,
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        user_agent=AgentServerRef(type="responses_api_agents", name="user"),
        assistant_agent=AgentServerRef(type="responses_api_agents", name="assistant"),
        judge_model=ModelServerRef(type="responses_api_models", name="support"),
        summary_model=ModelServerRef(type="responses_api_models", name="support"),
        resources_tool_transports=["direct_http"],
        max_turns=2,
    )
    return UserSimEnvironmentServer(config=config, server_client=client), client


def test_configured_model_name_resolves_agent_and_direct_model_targets() -> None:
    server, _ = _environment_server()

    assert _configured_model_name(server, "user_model") == "nvidia/example-assistant-model"
    assert _configured_model_name(server, "assistant_model") == "nvidia/example-assistant-model"
    assert _configured_model_name(server, "judge_model") == "nvidia/example-support-model"
    assert _configured_model_name(server, "summary_model") == "nvidia/example-support-model"


def _request() -> UserSimEpisodeRequest:
    return UserSimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="usersim:example", task_id="task"),
            task_input=UserSimTaskInput(
                sampling={"locale": "en_US", "seed": 42},
                responses_create_params={
                    "user": {"input": [], "temperature": 0.8},
                    "assistant": {"input": [], "temperature": 0.2},
                    "judge": {"input": []},
                    "summary": {"input": []},
                },
            ),
        ),
    )


def _queue_success_responses(client: _Client) -> None:
    client.responses.extend(
        [
            _Response(
                {
                    "resources_session_id": "resources-session",
                    "scenario": {
                        "persona": {"first_name": "Morgan"},
                        "probe_type": "general_open_ended",
                        "theme": {"type": "recommendation", "description": "Plan dinner."},
                        "goal": "Plan dinner.",
                        "locale": "en_US",
                    },
                    "usersim_context": {
                        "locale": "en_US",
                        "seed": 42,
                        "personas_dataset_version": "0.0.2",
                        "personas_panel_sha256": "a" * 64,
                        "usersim_revision": "b" * 40,
                    },
                },
                cookie="resources-cookie",
            ),
            _Response({"agent_session_id": "user-session"}, cookie="user-cookie"),
            _Response({"agent_session_id": "assistant-session"}, cookie="assistant-cookie"),
            _Response(_model_response("user-response", "I need dinner advice.")),
            _Response(_model_response("assistant-response", "Try a lentil curry.")),
            _Response(_model_response("judge-response", "<rating>pass</rating>")),
            _Response(_model_response("summary-response", "The assistant recommended lentil curry.")),
            _Response(
                {
                    "agent_session_id": "assistant-session",
                    "resources_cookies": {"session": "resources-cookie"},
                }
            ),
            _Response(
                {
                    "agent_session_id": "user-session",
                    "resources_cookies": {},
                }
            ),
            _Response(
                {
                    "reward": 1.0,
                    "reward_components": {
                        "participants_completed": 1.0,
                        "shared_state_exercised": 0.0,
                        "terminated": 0.0,
                    },
                    "scenario_completed": True,
                    "verifier_data": {},
                }
            ),
            _Response({"resources_session_id": "resources-session"}),
        ]
    )


def test_usersim_generator_uses_instance_api() -> None:
    config = object()
    models = {"user_model": object()}

    class Generator:
        @property
        def config(self) -> object:
            return self._config

        def generate(self, data: dict[str, Any]) -> dict[str, Any]:
            return {
                "config": self.config,
                "model": self.get_model("user_model"),
                "data": data,
            }

    generator = _create_usersim_generator(Generator, config, models)

    assert isinstance(generator, Generator)
    assert generator.generate({"probe_type": "general_open_ended"}) == {
        "config": config,
        "model": models["user_model"],
        "data": {"probe_type": "general_open_ended"},
    }


async def test_tool_probe_uses_remote_driver_and_preserves_tool_transcript(monkeypatch) -> None:
    environment_server, _ = _environment_server()
    synchronized = []

    async def synchronize(_bridge, transcript):
        synchronized.append([dict(message) for message in transcript])

    monkeypatch.setattr(environment_server, "_synchronize_runtime", synchronize)
    monkeypatch.setattr(
        "environment_servers.usersim.app._create_usersim_generator",
        lambda *_args, **_kwargs: pytest.fail("tool probes must not construct a local UserSim generator"),
    )

    response = NeMoGymResponse.model_validate(_tool_model_response())

    class Bridge:
        resources_cookies = {"session": "resources"}
        invocations = []

        async def invoke(self, alias, messages, *, max_tokens, tools):
            assert alias == "assistant_model"
            assert tools and tools[0]["function"]["name"] == "safe_action"
            self.invocations.append(SimpleNamespace(response=response))
            return SimpleNamespace(message=SimpleNamespace(content="Done."))

    seed = UserSimSeedResponse.model_validate(
        {
            "resources_session_id": "resources-session",
            "scenario": {
                "persona": {"first_name": "Morgan"},
                "probe_type": "safety_agentic",
                "theme": "safety",
            },
            "usersim_context": {
                "locale": "en_US",
                "seed": 42,
                "personas_dataset_version": "0.0.2",
                "personas_panel_sha256": "a" * 64,
                "usersim_revision": "b" * 40,
            },
            "runtime_descriptor": _runtime_descriptor().model_dump(mode="json"),
        }
    )
    result = await environment_server._run_usersim(Bridge(), seed)

    assert [message["role"] for message in result["conversation_messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result["conversation_messages"][1]["tool_calls"][0]["id"] == "call-1"
    assert result["conversation_messages"][2]["tool_call_id"] == "call-1"
    assert synchronized[-1] == result["conversation_messages"]


async def test_usersim_environment_server_runs_native_episode(monkeypatch) -> None:
    environment_server, client = _environment_server()
    _queue_success_responses(client)

    async def fake_run(bridge, _scenario):
        await bridge.invoke("user_model", [{"role": "user", "content": "write user"}], max_tokens=None, tools=None)
        await bridge.invoke(
            "assistant_model",
            [{"role": "user", "content": "I need dinner advice."}],
            max_tokens=128,
            tools=None,
        )
        await bridge.invoke(
            "judge_model",
            [{"role": "user", "content": "judge"}],
            max_tokens=None,
            tools=None,
        )
        await bridge.invoke(
            "summary_model",
            [{"role": "user", "content": "summarize"}],
            max_tokens=None,
            tools=None,
        )
        return {
            "conversation_messages": [
                {"role": "user", "content": "I need dinner advice."},
                {"role": "assistant", "content": "Try a lentil curry."},
            ],
            "conversation_status": True,
            "simulation_outcome": {"status": "ok", "early_stop": True},
            "simulation_traces": [],
        }

    monkeypatch.setattr(environment_server, "_run_usersim", fake_run)
    response = await environment_server.run_request(_request())

    assert isinstance(environment_server, BaseEnvironmentServer)
    assert response.failure is None
    assert response.result is not None
    assert response.result.verification.reward == 1.0
    assert [invocation.role for invocation in response.result.invocations] == ["user", "assistant", "judge", "summary"]
    assert response.result.invocations[1].request.max_output_tokens == 128
    assert response.result.invocations[1].termination_reason == "usersim_early_stop"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/responses",
        "/v1/responses",
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert client.calls[1][2]["json"]["tool_accesses"] == []
    [tool_access] = client.calls[2][2]["json"]["tool_accesses"]
    assert tool_access["name"] == "resources.direct_http"
    assert tool_access["cookies"] == {"session": "resources-cookie"}
    assert client.calls[3][2]["cookies"] == {"session": "user-cookie"}
    assert client.calls[4][2]["cookies"] == {"session": "assistant-cookie"}
    assert client.calls[5][0] == "support"
    assert client.calls[6][0] == "support"
    assert "cookies" not in client.calls[5][2]
    assert "cookies" not in client.calls[6][2]
    verify_body = client.calls[9][2]["json"]
    assert verify_body.task_id == TaskId(taskset="usersim:example", task_id="task")
    assert [invocation.role for invocation in verify_body.verification_input.invocations] == [
        "user",
        "assistant",
        "judge",
        "summary",
    ]


async def test_token_capture_uses_environment_episode_identity(monkeypatch) -> None:
    environment_server, client = _environment_server(token_capture=True)
    _queue_success_responses(client)

    async def fake_run(bridge, _scenario):
        await bridge.invoke("user_model", [{"role": "user", "content": "write user"}], max_tokens=None, tools=None)
        await bridge.invoke(
            "assistant_model",
            [{"role": "user", "content": "hello"}],
            max_tokens=None,
            tools=None,
        )
        await bridge.invoke(
            "judge_model",
            [{"role": "user", "content": "judge"}],
            max_tokens=None,
            tools=None,
        )
        await bridge.invoke(
            "summary_model",
            [{"role": "user", "content": "summarize"}],
            max_tokens=None,
            tools=None,
        )
        return {
            "conversation_messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer"},
            ],
            "conversation_status": True,
            "simulation_outcome": {},
        }

    monkeypatch.setattr(environment_server, "_run_usersim", fake_run)
    await environment_server.run_request(_request())

    for call_index in (3, 4):
        assert client.calls[call_index][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"
    for call_index in (5, 6):
        assert client.calls[call_index][1] == "/v1/responses"


def test_dependency_retry_requires_transient_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ValueError("invalid contract")) is False
