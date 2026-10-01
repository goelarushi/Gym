# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import orjson
import pytest
from omegaconf import OmegaConf
from pydantic import ConfigDict

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _apply_activation_parameters,
    _ConversationBridge,
    _is_retryable_dependency_error,
    _response_chat_message,
    _to_responses_input_items,
)
from nemo_gym.base_environment_server import BaseEnvironmentServer
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from resources_servers.usersim.app import UserSimResourcesServer
from resources_servers.usersim.episode_contracts import (
    ProbeRuntimeDescriptor,
    UserSimActivationRequest,
    UserSimEpisodeRequest,
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
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 15,
        },
    }


def _tool_model_response() -> dict[str, Any]:
    response = _model_response("assistant-response", "Done.")
    response["output"] = [
        {
            "id": "reasoning-1",
            "summary": [{"text": "Inspect the tool result.", "type": "summary_text"}],
            "type": "reasoning",
        },
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
        }
    )


def test_tool_transcript_converts_to_responses_input_items() -> None:
    assistant_items = _to_responses_input_items(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "safe_action", "arguments": '{"value":"x"}'},
                }
            ],
        }
    )
    tool_items = _to_responses_input_items({"role": "tool", "content": '{"ok":true}', "tool_call_id": "call-1"})

    assert assistant_items == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "safe_action",
            "arguments": '{"value":"x"}',
        }
    ]
    assert tool_items == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"ok":true}',
        }
    ]


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
    )
    return UserSimEnvironmentServer(config=config, server_client=client), client


def test_shipped_usersim_servers_disable_ray() -> None:
    assert UserSimResourcesServer.ray_enabled is False
    assert UserSimEnvironmentServer.ray_enabled is False


def _request() -> UserSimEpisodeRequest:
    return UserSimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="usersim:example", task_id="task"),
            task_input=UserSimTaskInput(
                resolved_row={
                    "persona": {"first_name": "Morgan"},
                    "probe_type": "general_open_ended",
                    "probe_family": "general_open_ended",
                    "probe_variant": "default",
                    "theme": {"type": "recommendation", "description": "Plan dinner."},
                    "locale": "en_US",
                    "trajectory_id": "native-trajectory",
                    "usersim_provenance": {
                        "code_sha": "a4665b3ce1a030e83871232e2fb69e5b39480818",
                    },
                    "usersim_config": {"random_seed": 42},
                },
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
                    "resolved_row": {
                        "persona": {"first_name": "Morgan"},
                        "probe_type": "general_open_ended",
                        "probe_family": "general_open_ended",
                        "probe_variant": "default",
                        "theme": {"type": "recommendation", "description": "Plan dinner."},
                        "locale": "en_US",
                        "trajectory_id": "native-trajectory",
                        "usersim_provenance": {
                            "code_sha": "a4665b3ce1a030e83871232e2fb69e5b39480818",
                        },
                        "usersim_config": {"random_seed": 42},
                    },
                    "assistant_tools": _runtime_descriptor().assistant_tools,
                    "runtime_descriptor": _runtime_descriptor().model_dump(mode="json"),
                },
                cookie="resources-cookie",
            ),
            _Response({"agent_session_id": "user-session"}, cookie="user-cookie"),
            _Response({"agent_session_id": "assistant-session"}, cookie="assistant-cookie"),
            _Response(
                {
                    "activation_id": "activation-000001",
                    "role": "user",
                    "model_alias": "user_model",
                    "messages": [{"role": "user", "content": "write user"}],
                    "parameters": {},
                }
            ),
            _Response(_model_response("user-response", "I need dinner advice.")),
            _Response(
                {
                    "activation_id": "activation-000002",
                    "role": "assistant",
                    "model_alias": "assistant_model",
                    "messages": [{"role": "user", "content": "I need dinner advice."}],
                    "parameters": {"max_tokens": 128, "temperature": 0.3, "top_p": 0.9},
                    "tools": _runtime_descriptor().assistant_tools,
                    "tools_enabled": True,
                    "continues_turn": False,
                }
            ),
            _Response(_model_response("assistant-response", "Try a lentil curry.")),
            _Response(
                {
                    "activation_id": "activation-000003",
                    "role": "judge",
                    "model_alias": "judge_model",
                    "messages": [{"role": "user", "content": "judge"}],
                    "parameters": {},
                }
            ),
            _Response(_model_response("judge-response", "<rating>pass</rating>")),
            _Response(
                {
                    "activation_id": "activation-000004",
                    "role": "summary",
                    "model_alias": "summary_model",
                    "messages": [{"role": "user", "content": "summarize"}],
                    "parameters": {},
                }
            ),
            _Response(_model_response("summary-response", "The assistant recommended lentil curry.")),
            _Response(
                {
                    "complete": True,
                    "result": {
                        "conversation_messages": [
                            {"role": "user", "content": "I need dinner advice."},
                            {"role": "assistant", "content": "Try a lentil curry."},
                        ],
                        "conversation_status": True,
                        "simulation_outcome": {"status": "ok", "early_stop": True},
                        "simulation_traces": [],
                    },
                }
            ),
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


def test_assistant_response_keeps_tool_calls_and_reasoning() -> None:
    """UserSim stores reasoning by default; dropping it here would lose it."""
    message = _response_chat_message(NeMoGymResponse.model_validate(_tool_model_response()))

    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == "call-1"
    assert message["reasoning_content"] == "Inspect the tool result."


def test_strict_response_format_requires_all_nullable_fields_and_forbids_extras() -> None:
    values: dict[str, Any] = {}
    _apply_activation_parameters(
        values,
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "safety_agentic_judgment",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "anyOf": [{"enum": [1, 2, 3, 4, 5], "type": "integer"}, {"type": "null"}],
                                "default": None,
                            },
                            "reasoning": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                        },
                    },
                },
            }
        },
        tools=[],
    )

    schema = values["text"]["format"]["schema"]
    assert values["text"]["format"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["score", "reasoning"]
    assert "default" not in schema["properties"]["score"]
    assert "default" not in schema["properties"]["reasoning"]


async def test_conversation_bridge_rejects_role_alias_mismatch_before_invocation() -> None:
    environment_server, client = _environment_server()
    request = _request()
    bridge = _ConversationBridge(
        environment_server,
        request,
        request.task.task_input,
        {},
        {},
        [],
    )

    with pytest.raises(ValueError, match="does not match alias"):
        await bridge.invoke(
            UserSimActivationRequest(
                activation_id="activation-role-mismatch",
                role="judge",
                model_alias="assistant_model",
                messages=[{"role": "user", "content": "Respond as the assistant."}],
                parameters={},
            )
        )

    assert client.calls == []


async def test_usersim_environment_server_runs_native_episode() -> None:
    environment_server, client = _environment_server()
    _queue_success_responses(client)
    response = await environment_server.run_request(_request())

    assert isinstance(environment_server, BaseEnvironmentServer)
    assert response.failure is None
    assert response.result is not None
    assert response.result.verification.reward == 1.0
    assert [invocation.role for invocation in response.result.invocations] == ["user", "assistant", "judge", "summary"]
    assert response.result.invocations[1].request.max_output_tokens == 128
    assert response.result.invocations[1].request.temperature == 0.3
    assert response.result.invocations[1].request.top_p == 0.9
    assert response.result.invocations[1].response.usage.total_tokens == 15
    assert response.result.invocations[1].termination_reason == "usersim_early_stop"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions",
        "/runtime/start",
        "/ng-rollout/rollout-a2/v1/responses",
        "/runtime/advance",
        # The Assistant Agent recorded its own turn, so this server re-reads
        # where the episode got to instead of submitting a result.
        "/ng-rollout/rollout-a2/v1/responses",
        "/runtime/pending",
        "/v1/responses",
        "/runtime/advance",
        "/v1/responses",
        "/runtime/advance",
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert client.calls[1][2]["json"]["tool_accesses"] == []
    # The Assistant Agent keeps Resources access: it records each response and
    # collects UserSim's tool payloads itself.
    [tool_access] = client.calls[2][2]["json"]["tool_accesses"]
    assert tool_access["name"] == "resources.direct_http"
    assert tool_access["cookies"] == {"session": "resources-cookie"}
    assert client.calls[2][2]["json"]["record_outputs_path"] == "/runtime/record"
    assert client.calls[1][2]["json"]["record_outputs_path"] is None
    assert client.calls[4][2]["cookies"] == {"session": "user-cookie"}
    assert client.calls[6][2]["cookies"] == {"session": "assistant-cookie"}
    assert client.calls[8][0] == "support"
    assert client.calls[10][0] == "support"
    assert "cookies" not in client.calls[8][2]
    assert "cookies" not in client.calls[10][2]
    assert client.calls[0][2]["json"].task_data == _request().task.task_input.model_dump(mode="json")
    verify_body = client.calls[14][2]["json"]
    assert verify_body.task_id == TaskId(taskset="usersim:example", task_id="task")
    assert verify_body.verification_input.resolved_row["trajectory_id"] == "native-trajectory"
    assert [invocation.role for invocation in verify_body.verification_input.invocations] == [
        "user",
        "assistant",
        "judge",
        "summary",
    ]


async def test_token_capture_uses_environment_episode_identity() -> None:
    environment_server, client = _environment_server(token_capture=True)
    _queue_success_responses(client)
    await environment_server.run_request(_request())

    for call_index in (4, 6):
        assert client.calls[call_index][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"
    for call_index in (8, 10):
        assert client.calls[call_index][1] == "/v1/responses"


def test_dependency_retry_requires_transient_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ValueError("invalid contract")) is False
