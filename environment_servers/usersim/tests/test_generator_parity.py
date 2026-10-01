# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity between a UserSim-prepared task and Gym's external execution path."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import ConfigDict
from usersim.engine.core.episode_runtime import HostRoleModel, ProbeEpisodeRuntime
from usersim.engine.core.probes import known_probes
from usersim.engine.external import materialize_episode_inputs
from usersim.engine.generator import ConversationSimulatorGenerator

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _apply_activation_parameters,
    _to_responses_input_items,
)
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, BaseServerConfig, ServerClient
from resources_servers.usersim.app import UserSimResourcesServer, UserSimResourcesServerConfig
from resources_servers.usersim.episode_contracts import (
    UserSimActivationRequest,
    UserSimActivationResult,
    UserSimAgentRecordRequest,
    UserSimRoleModel,
    UserSimSeedSessionRequest,
)
from responses_api_agents.simple_agent.app import TOOL_CALL_ID_HEADER, SimpleAgent, SimpleAgentConfig


USERSIM_REVISION = "a4665b3ce1a030e83871232e2fb69e5b39480818"  # pragma: allowlist secret


def _message_value(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _arguments_for_schema(schema: dict[str, Any]) -> Any:
    if enum := schema.get("enum"):
        return enum[0]
    if schema.get("type") == "object":
        required = set(schema.get("required", []))
        return {
            name: _arguments_for_schema(value)
            for name, value in schema.get("properties", {}).items()
            if name in required
        }
    return {
        "array": [],
        "boolean": True,
        "integer": 1,
        "number": 1,
        "string": "example",
    }.get(schema.get("type"), "example")


class _ReplayableModel:
    model_name = "nvidia/nemotron-3-super-120b-a12b"

    def __init__(self, role: str) -> None:
        self.role = role

    async def acompletion(self, messages: list[Any], **kwargs: Any) -> SimpleNamespace:
        tool_calls = None
        has_tool_result = any(str(_message_value(message, "role")) == "tool" for message in messages)
        if self.role == "assistant" and kwargs.get("tools") and not has_tool_result:
            function = kwargs["tools"][0]["function"]
            tool_calls = [
                {
                    "id": "call-parity",
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(_arguments_for_schema(function.get("parameters", {})), sort_keys=True),
                    },
                }
            ]
            content = ""
        else:
            content = {
                "api": '{"ok":true}',
                "assistant": "Scripted assistant response.",
                "judge": "<explanation>valid scripted turn</explanation><rating>success</rating>",
                "summary": "yes",
                "user": "Could you explain that a little more?",
            }[self.role]
        return SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                reasoning_content=f"{self.role} reasoning",
                tool_calls=tool_calls,
            ),
            usage=None,
        )


def _chat_response(response: SimpleNamespace) -> dict[str, Any]:
    result = {
        "role": "assistant",
        "content": response.message.content or "",
        "reasoning_content": response.message.reasoning_content,
        "tool_calls": deepcopy(response.message.tool_calls),
    }
    return {name: value for name, value in result.items() if value is not None}


class _HTTPResponse:
    ok = True
    status = 200
    cookies: dict[str, Any] = {}

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.content = self

    async def read(self) -> bytes:
        return self.body


class _ParityClient(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    resources: UserSimResourcesServer
    resources_request: Any
    model_calls: int = 0

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _HTTPResponse:
        if server_name == "resources":
            if url_path == "/runtime/start":
                value = await self.resources.start_runtime_lifecycle(self.resources_request)
                return _HTTPResponse(value.model_dump_json().encode())
            if url_path == "/runtime/advance":
                value = await self.resources.advance_runtime_lifecycle(
                    self.resources_request,
                    UserSimActivationResult.model_validate(kwargs["json"]),
                )
                return _HTTPResponse(value.model_dump_json().encode())
            if url_path == "/runtime/pending":
                value = await self.resources.pending_runtime_event(self.resources_request)
                return _HTTPResponse(value.model_dump_json().encode())
            if url_path == "/runtime/record":
                value = await self.resources.record_agent_output(
                    self.resources_request,
                    UserSimAgentRecordRequest.model_validate(kwargs["json"]),
                )
                return _HTTPResponse(value.model_dump_json().encode())
            response = await self.resources.invoke_probe_tool(
                self.resources_request,
                url_path.removeprefix("/"),
                kwargs["json"],
                kwargs["headers"][TOOL_CALL_ID_HEADER],
            )
            return _HTTPResponse(response.body)

        self.model_calls += 1
        body = kwargs["json"]
        has_tool_result = any(getattr(item, "type", None) == "function_call_output" for item in body.input)
        if body.tools and not has_tool_result:
            tool = body.tools[0]
            tool_name = _message_value(tool, "name")
            tool_parameters = _message_value(tool, "parameters", {})
            output = [
                {
                    "id": "reasoning-parity",
                    "summary": [{"text": "assistant reasoning", "type": "summary_text"}],
                    "type": "reasoning",
                },
                {
                    "id": "function-parity",
                    "call_id": "call-parity",
                    "name": tool_name,
                    "arguments": json.dumps(_arguments_for_schema(tool_parameters), sort_keys=True),
                    "type": "function_call",
                    "status": "completed",
                },
            ]
        else:
            output = [
                {
                    "id": "reasoning-parity",
                    "summary": [{"text": "assistant reasoning", "type": "summary_text"}],
                    "type": "reasoning",
                },
                {
                    "id": "message-parity",
                    "content": [{"annotations": [], "text": "Scripted assistant response.", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ]
        response = {
            "id": f"response-{self.model_calls}",
            "created_at": 1,
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        return _HTTPResponse(json.dumps(response).encode())

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


class _GymBridge:
    def __init__(self, agent: SimpleAgent, assistant_tools: list[dict[str, Any]]) -> None:
        self.agent = agent
        self.assistant_tools = assistant_tools
        self.resources_cookies: dict[str, str] = {}
        self.models = {role: _ReplayableModel(role) for role in ("user", "judge", "summary")}

    async def invoke(self, activation: UserSimActivationRequest) -> UserSimActivationResult:
        response = await self.models[activation.role].acompletion(
            activation.messages,
            **activation.parameters,
        )
        return UserSimActivationResult(
            activation_id=activation.activation_id,
            response=_chat_response(response),
        )

    async def invoke_assistant(self, activation: UserSimActivationRequest) -> None:
        values: dict[str, Any] = {
            "input": [item for message in activation.messages for item in _to_responses_input_items(message)]
        }
        _apply_activation_parameters(values, activation.parameters, tools=activation.tools)
        # The Agent owns the model/tool loop: it records each response with
        # Resources, which is what advances the episode, and follows the reply.
        await self.agent._create_episode(
            NeMoGymResponseCreateParamsNonStreaming.model_validate(values),
            model_url_path="/v1/responses",
            resources_server_cookies={},
            record_outputs_path="/runtime/record",
        )
        return None


def _normalized_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(result)
    for name in ("conversation_messages", "conversation_metadata", "simulation_outcome", "simulation_traces"):
        if isinstance(normalized.get(name), str):
            normalized[name] = json.loads(normalized[name])
    normalized["simulation_outcome"].pop("wall_clock_s", None)
    normalized["simulation_outcome"].pop("wall_clock_s_by_alias", None)
    return normalized


@pytest.mark.parametrize("probe_type", known_probes())
async def test_prepared_task_matches_standalone_generator_through_gym_stack(probe_type, monkeypatch) -> None:
    """Every registered probe reproduces a standalone run through the Gym stack."""
    monkeypatch.setattr("usersim.engine.core.episode_input.get_code_sha", lambda: USERSIM_REVISION)
    [resolved_row] = materialize_episode_inputs(
        locale="en_US",
        num_rows=1,
        probe_mix={probe_type: 1.0},
        random_seed=42,
    )
    direct_models = {
        "api_response_model": _ReplayableModel("api"),
        **{f"{role}_model": _ReplayableModel(role) for role in ("assistant", "judge", "summary", "user")},
    }
    direct_runtime = ProbeEpisodeRuntime.from_resolved_row(deepcopy(resolved_row), models=direct_models)
    provider = SimpleNamespace(
        model_registry=SimpleNamespace(get_model=lambda *, model_alias: direct_models[model_alias])
    )
    standalone = await ConversationSimulatorGenerator(direct_runtime.config, provider).agenerate(
        deepcopy(resolved_row)
    )

    resources = UserSimResourcesServer(
        config=UserSimResourcesServerConfig(
            host="resources",
            port=8000,
            entrypoint="app.py",
            name="resources",
            tool_simulation_model=ModelServerRef(type="responses_api_models", name="support"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    # The API-response simulator is UserSim's own, in both runs, and the
    # declared role models are the ones the standalone run resolved.
    monkeypatch.setattr(
        UserSimResourcesServer,
        "_create_probe_runtime",
        lambda _self, resolved, role_models: ProbeEpisodeRuntime.from_resolved_row(
            resolved,
            models={
                "api_response_model": _ReplayableModel("api"),
                **{
                    alias: HostRoleModel(model_name=model.model_name, max_tokens=model.max_tokens)
                    for alias, model in role_models.items()
                },
            },
        ),
    )
    resources_request = SimpleNamespace(session={SESSION_ID_KEY: "parity-session"})
    seed = await resources.seed_session(
        resources_request,
        UserSimSeedSessionRequest(
            resources_session_id="resources-session",
            episode_id=EpisodeId(rollout_id="parity", attempt=0),
            task_id=TaskId(taskset="usersim:example", task_id=probe_type),
            task_data={"resolved_row": resolved_row},
            role_models={
                alias: UserSimRoleModel(model_name=_ReplayableModel.model_name)
                for alias in ("user_model", "assistant_model", "judge_model", "summary_model")
            },
        ),
    )
    client = _ParityClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        resources=resources,
        resources_request=resources_request,
    )
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="assistant",
            port=8001,
            entrypoint="app.py",
            name="assistant",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            model_server=ModelServerRef(type="responses_api_models", name="model"),
        ),
        server_client=client,
    )
    environment = UserSimEnvironmentServer(
        config=UserSimEnvironmentServerConfig(
            host="environment",
            port=8002,
            entrypoint="app.py",
            name="environment",
            cleanup_timeout_seconds=10,
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            user_agent=AgentServerRef(type="responses_api_agents", name="user"),
            assistant_agent=AgentServerRef(type="responses_api_agents", name="assistant"),
            judge_model=ModelServerRef(type="responses_api_models", name="judge"),
            summary_model=ModelServerRef(type="responses_api_models", name="summary"),
            resources_tool_transports=["direct_http"],
        ),
        server_client=client,
    )
    gym_result = await environment._run_usersim(_GymBridge(agent, seed.assistant_tools))

    standalone_result = {name: standalone[name] for name in gym_result}
    assert _normalized_result(gym_result) == _normalized_result(standalone_result)
    assert resolved_row["trajectory_id"] == standalone["trajectory_id"]
    assert resolved_row["usersim_provenance"] == standalone["usersim_provenance"]
    assert client.model_calls >= 1
