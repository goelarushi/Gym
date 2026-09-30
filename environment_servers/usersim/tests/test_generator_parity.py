# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity between a UserSim-prepared task and Gym's external execution path."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from omegaconf import OmegaConf
from pydantic import ConfigDict
from usersim.engine.core.episode_runtime import ProbeEpisodeRuntime
from usersim.engine.external import materialize_episode_inputs
from usersim.engine.generator import ConversationSimulatorGenerator

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _apply_activation_parameters,
    _response_output_messages,
    _to_responses_input_items,
)
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.base_responses_api_agent import AgentToolLoopPolicy
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, BaseServerConfig, ServerClient
from resources_servers.usersim.app import UserSimResourcesServer, UserSimResourcesServerConfig
from resources_servers.usersim.episode_contracts import UserSimActivationRequest, UserSimActivationResult
from responses_api_agents.simple_agent.app import TOOL_CALL_ID_HEADER, SimpleAgent, SimpleAgentConfig


USERSIM_REVISION = "4fd4c800bbef8883329543df632f328860fc6429"


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
            if url_path == "/runtime/tool_calls":
                value = await self.resources.invoke_probe_tool_batch(self.resources_request, kwargs["json"])
                return _HTTPResponse(json.dumps(value).encode())
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
        if activation.role != "assistant":
            response = await self.models[activation.role].acompletion(
                activation.messages,
                **activation.parameters,
            )
            return UserSimActivationResult(
                activation_id=activation.activation_id,
                response=_chat_response(response),
            )

        values: dict[str, Any] = {
            "input": [item for message in activation.messages for item in _to_responses_input_items(message)]
        }
        _apply_activation_parameters(values, activation.parameters, assistant_tools=self.assistant_tools)
        response, _, _, _ = await self.agent._create_episode(
            NeMoGymResponseCreateParamsNonStreaming.model_validate(values),
            model_url_path="/v1/responses",
            resources_server_cookies={},
            tool_loop_policy=(
                AgentToolLoopPolicy.model_validate(activation.assistant_tool_loop_policy.model_dump(mode="json"))
                if activation.assistant_tool_loop_policy is not None
                else None
            ),
            tool_batch_path="/runtime/tool_calls",
        )
        return UserSimActivationResult(
            activation_id=activation.activation_id,
            transcript_delta=_response_output_messages(NeMoGymResponse.model_validate(response)),
        )


def _normalized_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(result)
    for name in ("conversation_messages", "conversation_metadata", "simulation_outcome", "simulation_traces"):
        if isinstance(normalized.get(name), str):
            normalized[name] = json.loads(normalized[name])
    normalized["simulation_outcome"].pop("wall_clock_s", None)
    normalized["simulation_outcome"].pop("wall_clock_s_by_alias", None)
    return normalized


async def test_prepared_task_matches_standalone_generator_through_gym_stack(monkeypatch) -> None:
    monkeypatch.setattr("usersim.engine.core.episode_input.get_code_sha", lambda: USERSIM_REVISION)
    [resolved_row] = materialize_episode_inputs(
        locale="en_US",
        num_rows=1,
        probe_mix={"safety_agentic": 1.0},
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
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    resources_request = SimpleNamespace(session={SESSION_ID_KEY: "parity-session"})
    seed = await resources.seed_session(
        resources_request,
        ResourcesSeedSessionRequest(
            resources_session_id="resources-session",
            episode_id=EpisodeId(rollout_id="parity", attempt=0),
            task_id=TaskId(taskset="usersim:example", task_id="safety_agentic"),
            task_data={"resolved_row": resolved_row},
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
    traces = _normalized_result(gym_result)["simulation_traces"]
    assert [(trace["turn_idx"], trace["call_idx"]) for trace in traces] == [(0, 0)]
    assert client.model_calls == 2
