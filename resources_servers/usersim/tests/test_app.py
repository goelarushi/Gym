# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.usersim.app import UserSimResourcesServer, UserSimResourcesServerConfig


USERSIM_REVISION = "4fd4c800bbef8883329543df632f328860fc6429"


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
    return SimpleNamespace(
        to_dict=lambda: {
            "probe_type": "safety_agentic",
            "assistant_tools": assistant_tools,
            "allowed_tool_names": ["safe_action"] if tools else [],
            "initial_user_message": "Do the task.",
            "loop_policy": {
                "tool_round_mode": "multi",
                "max_assistant_activations": 2,
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


def _runtime(*, tools: bool = False) -> MagicMock:
    runtime = MagicMock()
    runtime.descriptor = AsyncMock(return_value=_descriptor(tools=tools))
    runtime.evidence = AsyncMock(return_value={"result_extras": {}})
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


def test_tool_http_preserves_plain_text_and_delegates_native_batch_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tools=True)
    runtime.simulate_tool_call = AsyncMock(return_value="plain safety payload")
    runtime.simulate_tool_calls = AsyncMock(return_value=["first", "second"])
    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", lambda *_args, **_kwargs: runtime)

    with TestClient(_app().setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed_body()).status_code == 200
        single = client.post(
            "/safe_action",
            json={"value": "one"},
            headers={"X-NeMo-Gym-Tool-Call-Id": "call-1"},
        )
        batch_body = [
            {"tool_call_id": "call-2", "tool_name": "safe_action", "arguments": {"value": "two"}},
            {"tool_call_id": "call-3", "tool_name": "safe_action", "arguments": {"value": "three"}},
        ]
        batch = client.post("/runtime/tool_calls", json=batch_body)

    assert single.content == b"plain safety payload"
    assert single.text == "plain safety payload"
    assert single.headers["content-type"].startswith("text/plain")
    assert batch.json() == ["first", "second"]
    runtime.simulate_tool_call.assert_awaited_once_with(
        "safe_action",
        {"value": "one"},
        tool_call_id="call-1",
    )
    runtime.simulate_tool_calls.assert_awaited_once_with(batch_body)


def test_plain_text_tool_payload_completes_lifecycle_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tools=True)
    runtime.simulate_tool_call = AsyncMock(return_value="plain safety payload")
    activation = SimpleNamespace(
        to_dict=lambda: {
            "activation_id": "activation-1",
            "role": "assistant",
            "model_alias": "assistant_model",
            "messages": [{"role": "user", "content": "Do the task."}],
            "parameters": {},
            "assistant_tool_loop_policy": _descriptor(tools=True).to_dict()["loop_policy"],
        }
    )
    complete = SimpleNamespace(
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
    runtime.advance = AsyncMock(side_effect=[activation, complete])
    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", lambda *_args, **_kwargs: runtime)

    with TestClient(_app().setup_webserver()) as client:
        client.post("/seed_session", json=_seed_body())
        started = client.post("/runtime/start", json={})
        tool = client.post(
            "/safe_action",
            json={},
            headers={"X-NeMo-Gym-Tool-Call-Id": "call-1"},
        )
        completed = client.post(
            "/runtime/advance",
            json={
                "activation_id": started.json()["activation_id"],
                "transcript_delta": [
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
                    {"role": "tool", "content": tool.text, "tool_call_id": "call-1"},
                    {"role": "assistant", "content": "Done."},
                ],
            },
        )

    assert completed.status_code == 200
    assert completed.json()["complete"] is True
    runtime.simulate_tool_call.assert_awaited_once()
    assert runtime.advance.await_count == 2


def test_close_session_cancels_owned_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()
    lifecycle_task = MagicMock()
    lifecycle_task.done.return_value = True
    runtime._lifecycle_task = lifecycle_task
    monkeypatch.setattr(UserSimResourcesServer, "_create_probe_runtime", lambda *_args, **_kwargs: runtime)

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


def test_native_runtime_assigns_semantic_indices_for_ordered_parallel_calls() -> None:
    from usersim.engine.external import ProbeEpisodeRuntime

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
        config=SimpleNamespace(random_seed=42, max_turns=3),
        data={"user_interaction_style": "direct"},
        profile={"patience": 0.75},
    )
    tool_name = next(iter(runtime.allowed_tool_names))
    calls = [
        {"tool_call_id": "call-1", "tool_name": tool_name, "arguments": {"value": "first"}},
        {"tool_call_id": "call-2", "tool_name": tool_name, "arguments": {"value": "second"}},
    ]

    async def run_calls() -> dict:
        await runtime.simulate_tool_calls(calls)
        return await runtime.evidence()

    import asyncio

    evidence = asyncio.run(run_calls())
    actions = evidence["conversation_metadata"]["attempted_actions"]
    traces = evidence["simulation_traces"]
    assert [action["turn_idx"] for action in actions] == [0, 0]
    assert [(trace["turn_idx"], trace["call_idx"]) for trace in traces] == [(0, 0), (0, 1)]
