# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.usersim.app import UserSimResourcesServer, UserSimResourcesServerConfig


USERSIM_REVISION = "a4665b3ce1a030e83871232e2fb69e5b39480818"


def _resolved_row(probe_type: str = "safety_agentic") -> dict:
    return {
        "probe_type": probe_type,
        "probe_family": probe_type,
        "probe_variant": "default",
        "persona": {"first_name": "Morgan", "age": 42},
        "locale": "en_US",
        "conversation_language": "English",
        "trajectory_id": f"usersim-{probe_type}",
        "usersim_provenance": {"code_sha": USERSIM_REVISION},
        "usersim_config": {"random_seed": 42},
    }


def _seed_body(row: dict | None = None) -> dict:
    return {
        "resources_session_id": "resources-session-0",
        "episode_id": {"rollout_id": "0-0", "attempt": 0},
        "task_id": {"taskset": "usersim:example", "task_id": "0"},
        "task_data": {"resolved_row": row or _resolved_row()},
    }


def _descriptor(*, tools: bool = False) -> SimpleNamespace:
    assistant_tools = (
        [
            {
                "type": "function",
                "function": {
                    "name": "safe_action",
                    "description": "Perform an action.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        if tools
        else []
    )
    return SimpleNamespace(to_dict=lambda: {"probe_type": "safety_agentic", "assistant_tools": assistant_tools})


def _activation(*, tools: bool, continues_turn: bool, activation_id: str = "activation-1") -> SimpleNamespace:
    return SimpleNamespace(
        to_dict=lambda: {
            "activation_id": activation_id,
            "role": "assistant",
            "model_alias": "assistant_model",
            "messages": [{"role": "user", "content": "Do the task."}],
            "parameters": {},
            "tools": _descriptor(tools=True).to_dict()["assistant_tools"] if tools else [],
            "tools_enabled": tools,
            "continues_turn": continues_turn,
        }
    )


def _complete() -> SimpleNamespace:
    return SimpleNamespace(
        to_dict=lambda: {
            "complete": True,
            "result": {
                "conversation_messages": [
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
                    {"role": "tool", "content": "plain safety payload", "tool_call_id": "call-1"},
                    {"role": "assistant", "content": "Done."},
                ],
                "conversation_status": True,
                "simulation_outcome": {"status": "ok"},
            },
        }
    )


def _executed(call_id: str) -> SimpleNamespace:
    return SimpleNamespace(tool_call_id=call_id, payload="plain safety payload")


def _runtime(*, tools: bool = False) -> MagicMock:
    runtime = MagicMock()
    runtime.descriptor = AsyncMock(return_value=_descriptor(tools=tools))
    runtime.evidence = AsyncMock(return_value={"result_extras": {}})
    runtime.executed_tool_calls = AsyncMock(return_value=[])
    runtime.close = AsyncMock()
    return runtime


def _app() -> UserSimResourcesServer:
    config = UserSimResourcesServerConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="usersim",
    )
    return UserSimResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_resources_constructs_runtime_only_from_unchanged_resolved_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _resolved_row()
    runtime = _runtime()
    from_resolved_row = MagicMock(return_value=runtime)
    monkeypatch.setattr(
        "usersim.engine.core.episode_runtime.ProbeEpisodeRuntime.from_resolved_row",
        from_resolved_row,
    )

    with TestClient(_app().setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed_body(row))

    assert response.status_code == 200
    assert response.json()["resolved_row"] == row
    from_resolved_row.assert_called_once()
    assert from_resolved_row.call_args.args == (row,)
    assert set(from_resolved_row.call_args.kwargs) == {"models"}


def test_seed_rejects_wrong_revision_before_runtime_construction() -> None:
    row = _resolved_row()
    row["usersim_provenance"]["code_sha"] = "a" * 40

    with TestClient(_app().setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed_body(row))

    assert response.status_code == 422
    assert response.json()["detail"] == "Resolved row does not match the configured UserSim revision"


def test_declared_role_models_reach_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The models this environment runs, not preparation's defaults, key identity."""
    captured: dict[str, object] = {}

    def capture(_self, resolved_row, role_models):
        captured["role_models"] = role_models
        return _runtime()

    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", capture)
    body = _seed_body()
    body["role_models"] = {
        "assistant_model": {"model_name": "host/model-under-test", "max_tokens": 4096},
        "user_model": {"model_name": "host/support-model"},
    }

    with TestClient(_app().setup_webserver()) as client:
        assert client.post("/seed_session", json=body).status_code == 200

    role_models = captured["role_models"]
    assert role_models["assistant_model"].model_name == "host/model-under-test"
    assert role_models["assistant_model"].max_tokens == 4096
    assert role_models["user_model"].max_tokens is None


def test_record_tells_the_agent_to_continue_and_what_to_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Agent owns its loop but follows this reply instead of modelling it."""
    runtime = _runtime(tools=True)
    runtime.advance = AsyncMock(
        side_effect=[
            _activation(tools=True, continues_turn=False),
            _activation(tools=False, continues_turn=True, activation_id="activation-2"),
        ]
    )
    runtime.executed_tool_calls = AsyncMock(side_effect=[[], [_executed("call-1")]])
    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", lambda *_a, **_k: runtime)

    with TestClient(_app().setup_webserver()) as client:
        client.post("/seed_session", json=_seed_body())
        started = client.post("/runtime/start", json={})
        recorded = client.post(
            "/runtime/record",
            json={
                "response": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call-1", "type": "function", "function": {"name": "safe_action", "arguments": "{}"}}
                    ],
                },
                "usage": {"input_tokens": 12, "output_tokens": 5},
            },
        )

    assert started.json()["tools_enabled"] is True
    body = recorded.json()
    assert body["should_continue"] is True
    # Tools are off for the answer step, and the Agent is handed UserSim's own
    # next input rather than keeping its accumulated transcript.
    assert body["tools"] == []
    assert body["input"] == [{"type": "message", "role": "user", "content": "Do the task."}]
    assert body["executed_tool_call_ids"] == ["call-1"]
    # The Agent did not supply an activation id; the session knew which one.
    assert runtime.advance.await_args_list[1].args[0]["activation_id"] == "activation-1"


def test_record_stops_the_agent_when_the_turn_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(tools=True)
    runtime.advance = AsyncMock(side_effect=[_activation(tools=True, continues_turn=False), _complete()])
    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", lambda *_a, **_k: runtime)

    with TestClient(_app().setup_webserver()) as client:
        client.post("/seed_session", json=_seed_body())
        client.post("/runtime/start", json={})
        recorded = client.post(
            "/runtime/record",
            json={"response": {"role": "assistant", "content": "Done."}},
        )
        pending = client.post("/runtime/pending", json={})

    assert recorded.json()["should_continue"] is False
    assert recorded.json()["complete"] is True
    # The Environment re-reads where the episode got to after the Agent's turn.
    assert pending.json()["complete"] is True


def test_tool_route_returns_the_payload_usersim_produced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Payloads are UserSim's, verbatim; several probe responses are not JSON."""
    runtime = _runtime(tools=True)
    runtime.tool_result = AsyncMock(return_value="plain safety payload")
    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", lambda *_a, **_k: runtime)

    with TestClient(_app().setup_webserver()) as client:
        client.post("/seed_session", json=_seed_body())
        payload = client.post("/safe_action", json={}, headers={"X-NeMo-Gym-Tool-Call-Id": "call-1"})

    assert payload.text == "plain safety payload"
    assert payload.headers["content-type"].startswith("text/plain")
    runtime.tool_result.assert_awaited_once_with("call-1")


def test_contract_error_is_reported_to_the_host_not_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host bug must not be retried against, or blamed on, the model under test."""
    from usersim.engine.core.episode_runtime import EpisodeContractError

    runtime = _runtime(tools=True)
    runtime.advance = AsyncMock(
        side_effect=[_activation(tools=False, continues_turn=True), EpisodeContractError("offers no tools")]
    )
    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", lambda *_a, **_k: runtime)

    with TestClient(_app().setup_webserver()) as client:
        client.post("/seed_session", json=_seed_body())
        client.post("/runtime/start", json={})
        rejected = client.post(
            "/runtime/record",
            json={
                "response": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "late", "type": "function", "function": {"name": "safe_action", "arguments": "{}"}}
                    ],
                }
            },
        )

    assert rejected.status_code == 422
    assert "offers no tools" in rejected.json()["detail"]


def test_close_session_closes_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()
    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", lambda *_a, **_k: runtime)

    with TestClient(_app().setup_webserver()) as client:
        seed = client.post("/seed_session", json=_seed_body()).json()
        closed = client.post(
            "/close_session",
            json={
                "resources_session_id": seed["resources_session_id"],
                "episode_id": {"rollout_id": "0-0", "attempt": 0},
            },
        )

    assert closed.status_code == 200
    runtime.close.assert_awaited_once()


def test_native_runtime_assigns_semantic_indices_for_ordered_parallel_calls() -> None:
    """Parallel calls in one recorded response get UserSim's own indices."""
    import asyncio
    import json

    from usersim.engine.config import ConversationSimulatorConfig
    from usersim.engine.external import ActivationRequest, ActivationResult, ProbeEpisodeRuntime

    runtime = ProbeEpisodeRuntime(
        probe_type="safety_agentic",
        persona={
            "first_name": "Morgan",
            "last_name": "Lee",
            "age": 42,
            "city": "Seattle",
            "education_level": "Bachelor",
            "occupation": "Engineer",
        },
        locale="en_US",
        language="English",
        models={},
        config=ConversationSimulatorConfig(name="usersim_resources_test", random_seed=42, max_turns=3),
        data={"user_interaction_style": "direct"},
        profile={"patience": 0.75},
    )
    tool_name = runtime.assistant_tools[0]["function"]["name"]

    async def drive() -> list:
        event = await runtime.advance()
        issued = False
        while isinstance(event, ActivationRequest):
            if event.role == "assistant" and event.tools_enabled and not issued:
                issued = True
                response = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps({"value": value})},
                        }
                        for index, value in enumerate(("first", "second"), start=1)
                    ],
                }
            else:
                response = {"role": "assistant", "content": "Done."}
            event = await runtime.advance(ActivationResult(activation_id=event.activation_id, response=response))
        return await runtime.executed_tool_calls()

    executed = asyncio.run(drive())

    assert [(call.turn_idx, call.call_idx) for call in executed[:2]] == [(0, 0), (0, 1)]
    assert [call.tool_call_id for call in executed[:2]] == ["call-1", "call-2"]
