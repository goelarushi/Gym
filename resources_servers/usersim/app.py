# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NeMo UserSim scenario initialization backed by managed personas."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import (
    EpisodeId,
    TaskId,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from resources_servers.usersim.episode_contracts import (
    ProbeRuntimeDescriptor,
    UserSimActivationRequest,
    UserSimActivationResult,
    UserSimEpisodeLifecycleComplete,
    UserSimLifecycleEvent,
    UserSimSeedResponse,
    UserSimSimulationResult,
    UserSimTaskInput,
    UserSimVerification,
    UserSimVerifyRequest,
)
from resources_servers.usersim.response_format import responses_json_schema


PROBE_SCORERS = {
    "financial_services": "financial_services",
    "health_decision_support_disclosure": "health_disclosure_concealment",
    "health_general_disclosure": "health_disclosure_concealment",
    "health_therapy_disclosure": "health_disclosure_concealment",
    "health_triage_disclosure": "health_disclosure_concealment",
    "identity_disclosure": "identity_disclosure",
    "safety_agentic": "safety_agentic",
    "safety_chat_pressure": "safety_chat_pressure",
    "sov_ai_dynamic": "sov_ai_dynamic",
    "sov_ai_facts": "sov_ai_facts",
    "sov_ai_multilingual_parity": "sov_ai_multilingual_parity",
    "tool_calling": "tool_use",
}
ASSISTANT_QUALITY_AXES = ("helpfulness", "accuracy", "coherence")
logger = logging.getLogger(__name__)


class UserSimResourcesServerConfig(BaseResourcesServerConfig):
    usersim_revision: str = Field(
        "4fd4c800bbef8883329543df632f328860fc6429",  # pragma: allowlist secret
        pattern=r"^[0-9a-f]{40}$",
    )
    tool_simulation_model: ModelServerRef | None = None
    probe_scorer_model: ModelServerRef | None = None
    model_call_timeout_seconds: float = Field(300.0, gt=0)


class SeededUserSimEpisode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode_id: EpisodeId
    task_id: TaskId
    seed: UserSimSeedResponse
    runtime: Any | None = None


def _conversation_roles(result: UserSimSimulationResult) -> set[str]:
    messages = result.conversation_messages
    return {
        role for message in messages if isinstance(message, dict) and isinstance((role := message.get("role")), str)
    }


def _simulation_infrastructure_failure(result: UserSimSimulationResult) -> str | None:
    outcome = result.simulation_outcome
    if str(outcome.get("status", "")).lower() != "failed":
        return None
    attribution = str(outcome.get("failure_attribution", "")).lower()
    if attribution in {"assistant_model", "model_under_test"}:
        return None
    return str(outcome.get("failure_detail") or attribution or "UserSim simulation infrastructure failed")


class _ResourcesModelFacade:
    """Async UserSim facade backed by a Gym Model Server."""

    def __init__(
        self,
        server: "UserSimResourcesServer",
        model: ModelServerRef,
    ) -> None:
        self.server = server
        self.model = model
        self.model_name = model.name

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        supported = {
            "max_tokens",
            "max_completion_tokens",
            "parallel_tool_calls",
            "reasoning_effort",
            "response_format",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        }
        unsupported = set(kwargs) - supported
        if unsupported:
            raise NotImplementedError(f"Unsupported UserSim support-model options: {sorted(unsupported)}")
        try:
            async with asyncio.timeout(self.server.config.model_call_timeout_seconds):
                return await self._completion(messages, options=kwargs)
        except TimeoutError as error:
            raise TimeoutError(
                f"Timed out after {self.server.config.model_call_timeout_seconds}s waiting for {self.model.name}"
            ) from error

    async def _completion(
        self,
        messages: Sequence[Any],
        *,
        options: Mapping[str, Any],
    ) -> SimpleNamespace:
        params: dict[str, Any] = {
            "input": [item for message in messages for item in _to_responses_input_items(message)]
        }
        max_tokens = options.get("max_tokens") or options.get("max_completion_tokens")
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        for name in ("parallel_tool_calls", "temperature", "tool_choice", "top_p"):
            if options.get(name) is not None:
                params[name] = options[name]
        if options.get("reasoning_effort") is not None:
            params["reasoning"] = {"effort": options["reasoning_effort"]}
        if options.get("tools"):
            params["tools"] = [_to_responses_tool(tool) for tool in options["tools"]]
        response_format = options.get("response_format")
        if response_format is not None:
            json_schema = response_format.get("json_schema")
            if response_format.get("type") != "json_schema" or not isinstance(json_schema, Mapping):
                raise NotImplementedError(f"Unsupported response format: {response_format!r}")
            strict = json_schema.get("strict", True)
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema["name"],
                    "schema": responses_json_schema(json_schema["schema"], strict=strict),
                    "strict": strict,
                }
            }
        response = await self.server.server_client.post(
            server_name=self.model.name,
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming.model_validate(params),
        )
        await raise_for_status(response)
        gym_response = NeMoGymResponse.model_validate(await get_response_json(response))
        usage = gym_response.usage
        return SimpleNamespace(
            message=SimpleNamespace(
                content=_response_text(gym_response),
                reasoning_content=_response_reasoning(gym_response) or None,
                tool_calls=_response_tool_calls(gym_response) or None,
            ),
            usage=SimpleNamespace(**usage.model_dump(mode="python")) if usage is not None else None,
        )


class UserSimResourcesServer(SimpleResourcesServer):
    """Resolve one replayable persona and general-purpose probe per episode."""

    ray_enabled = False
    config: UserSimResourcesServerConfig
    session_id_to_seed: dict[str, SeededUserSimEpisode] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/runtime/start", response_model=UserSimLifecycleEvent)(self.start_runtime_lifecycle)
        app.post("/runtime/advance", response_model=UserSimLifecycleEvent)(self.advance_runtime_lifecycle)
        app.post("/runtime/tool_calls")(self.invoke_probe_tool_batch)
        app.post("/{tool_name}")(self.invoke_probe_tool)
        return app

    def _resolve_seed(self, task: UserSimTaskInput, resources_session_id: str) -> UserSimSeedResponse:
        provenance = task.resolved_row.get("usersim_provenance")
        if not isinstance(provenance, Mapping) or provenance.get("code_sha") != self.config.usersim_revision:
            raise HTTPException(status_code=422, detail="Resolved row does not match the configured UserSim revision")
        if not task.resolved_row.get("trajectory_id"):
            raise HTTPException(status_code=422, detail="Resolved row is missing UserSim trajectory identity")
        return UserSimSeedResponse(
            resources_session_id=resources_session_id,
            resolved_row=task.resolved_row,
        )

    def _seeded_episode(self, request: Request) -> SeededUserSimEpisode:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_seed:
            raise RuntimeError("No active NeMo UserSim scenario. Call /seed_session first.")
        return self.session_id_to_seed[session_id]

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest,
    ) -> UserSimSeedResponse:
        try:
            task = UserSimTaskInput.model_validate(body.task_data)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
        session_id = request.session[SESSION_ID_KEY]
        result = self._resolve_seed(task, body.resources_session_id)
        runtime = self._create_probe_runtime(result.resolved_row)
        descriptor = ProbeRuntimeDescriptor.model_validate((await runtime.descriptor()).to_dict())
        result = result.model_copy(
            update={
                "assistant_tools": descriptor.assistant_tools,
                "runtime_descriptor": descriptor,
            }
        )
        self.session_id_to_seed[session_id] = SeededUserSimEpisode(
            episode_id=body.episode_id,
            task_id=body.task_id,
            seed=result,
            runtime=runtime,
        )
        return result

    def _create_probe_runtime(
        self,
        resolved_row: dict[str, Any],
    ) -> Any:
        from usersim.engine.core.episode_runtime import ProbeEpisodeRuntime

        if resolved_row.get("probe_type") == "tool_calling" and self.config.tool_simulation_model is None:
            raise ValueError("tool_calling requires resources tool_simulation_model configuration")
        models = {}
        if self.config.tool_simulation_model is not None:
            models["api_response_model"] = _ResourcesModelFacade(
                self,
                self.config.tool_simulation_model,
            )
        return ProbeEpisodeRuntime.from_resolved_row(resolved_row, models=models)

    async def _score_native_result(
        self,
        seeded: SeededUserSimEpisode,
        native_result: UserSimSimulationResult,
    ) -> tuple[str | None, dict[str, Any] | None, bool]:
        resolved_row = seeded.seed.resolved_row
        probe_type = str(resolved_row["probe_type"])
        scorer_name = PROBE_SCORERS.get(probe_type)
        if (
            scorer_name == "health_disclosure_concealment"
            and resolved_row.get("probe_variant", "default") != "guarded"
        ):
            return None, None, True
        if scorer_name is None:
            return None, None, True

        result_extras = (await seeded.runtime.evidence())["result_extras"] if seeded.runtime is not None else {}
        trajectory = {
            **native_result.model_dump(mode="python"),
            **result_extras,
            **resolved_row,
        }
        scorer_models = {alias: model for alias, model in seeded.runtime.models.items() if alias == "judge_model"}
        if not scorer_models and self.config.probe_scorer_model is not None:
            scorer_models = {
                "judge_model": _ResourcesModelFacade(
                    self,
                    self.config.probe_scorer_model,
                )
            }

        try:
            from usersim.engine.evaluator.scorers import get_scorer

            scores = await get_scorer(scorer_name)(trajectory, scorer_models)
        except Exception as error:
            logger.exception("Native UserSim scorer %s failed", scorer_name)
            scores = {
                "status_proposal": False,
                "error": f"{type(error).__name__}: {error}",
            }
        passed = scores.get("status_proposal") is True and not scores.get("error")
        return scorer_name, scores, passed

    async def _evaluate_assistant_quality(
        self,
        seeded: SeededUserSimEpisode,
        native_result: UserSimSimulationResult,
    ) -> tuple[dict[str, Any], dict[str, float], float | None]:
        from usersim.engine.evaluator.runtime import TrajectoryEvaluatorRuntime
        from usersim.taxonomy.eval_cell import normalize_axis_score, score_from_eval_cell

        if self.config.probe_scorer_model is None:
            return (
                {
                    "axes": {},
                    "scorers": {},
                    "skipped": True,
                    "skipped_reason": "missing_probe_scorer_model",
                },
                {},
                None,
            )

        model = (
            seeded.runtime.models["judge_model"]
            if seeded.runtime is not None and "judge_model" in seeded.runtime.models
            else _ResourcesModelFacade(self, self.config.probe_scorer_model)
        )
        evaluator = TrajectoryEvaluatorRuntime(models={"judge_model": model})
        evaluation = await evaluator.evaluate(
            {
                **native_result.model_dump(mode="python"),
                **seeded.seed.resolved_row,
            }
        )
        normalized_scores: dict[str, float] = {}
        for axis in evaluation.get("envelope", {}).get("axes", []):
            score = score_from_eval_cell(evaluation, axis)
            if score is not None:
                normalized_scores[axis] = normalize_axis_score(axis, score)

        quality_scores = [normalized_scores.get(axis) for axis in ASSISTANT_QUALITY_AXES]
        assistant_quality = (
            sum(score for score in quality_scores if score is not None) / len(ASSISTANT_QUALITY_AXES)
            if all(score is not None for score in quality_scores)
            else None
        )
        return evaluation, normalized_scores, assistant_quality

    async def invoke_probe_tool(
        self,
        request: Request,
        tool_name: str,
        body: dict[str, Any] = Body(),
        tool_call_id: str | None = Header(None, alias="X-NeMo-Gym-Tool-Call-Id"),
    ) -> Response:
        """Simulate one tool selected for the request's seeded episode."""
        seeded = self._seeded_episode(request)
        if seeded.runtime is None:
            raise HTTPException(status_code=404, detail="This episode does not expose probe tools")
        if tool_call_id is None:
            raise HTTPException(status_code=422, detail="Runtime tool calls require a tool_call_id header")
        try:
            payload = await seeded.runtime.simulate_tool_call(
                tool_name,
                body,
                tool_call_id=tool_call_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return Response(content=payload, media_type="text/plain")

    async def invoke_probe_tool_batch(
        self,
        request: Request,
        body: list[dict[str, Any]] = Body(),
    ) -> list[str]:
        """Simulate one ordered parallel tool-call batch."""
        seeded = self._seeded_episode(request)
        if seeded.runtime is None:
            raise HTTPException(status_code=404, detail="This episode does not expose probe tools")
        try:
            return await seeded.runtime.simulate_tool_calls(body)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    async def start_runtime_lifecycle(
        self,
        request: Request,
    ) -> UserSimLifecycleEvent:
        """Start the canonical lifecycle and return its first activation."""
        seeded = self._seeded_episode(request)
        try:
            event = await seeded.runtime.advance()
        except (TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return _lifecycle_event(event)

    async def advance_runtime_lifecycle(
        self,
        request: Request,
        body: UserSimActivationResult,
    ) -> UserSimLifecycleEvent:
        """Submit one activation result and return the next lifecycle event."""
        seeded = self._seeded_episode(request)
        try:
            event = await seeded.runtime.advance(body.model_dump(mode="json"))
        except (TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return _lifecycle_event(event)

    async def verify(
        self,
        request: Request,
        body: UserSimVerifyRequest,
    ) -> UserSimVerification:
        seeded = self._seeded_episode(request)
        verification_input = body.verification_input
        if (
            body.episode_id != seeded.episode_id
            or body.task_id != seeded.task_id
            or verification_input.resolved_row != seeded.seed.resolved_row
        ):
            raise HTTPException(
                status_code=409,
                detail="Verified NeMo UserSim resolved episode does not match the seeded session",
            )
        native_result = verification_input.usersim_result
        if getattr(seeded.runtime, "_lifecycle_started", False):
            native_result = UserSimSimulationResult.model_validate(await seeded.runtime.finalize())
        native_scorer_name, native_scores, native_scorer_pass = await self._score_native_result(
            seeded,
            native_result,
        )
        assistant_eval, normalized_axis_scores, assistant_quality = await self._evaluate_assistant_quality(
            seeded,
            native_result,
        )
        participants_completed = {"user", "assistant"} <= _conversation_roles(native_result)
        scenario_completed = native_result.conversation_status and participants_completed and native_scorer_pass
        simulation_failure = _simulation_infrastructure_failure(native_result)
        scorer_failure = (
            str(native_scores.get("error")) if native_scores is not None and native_scores.get("error") else None
        )
        evaluator_failure = (
            str(assistant_eval.get("skipped_reason", "trajectory evaluator unavailable"))
            if assistant_quality is None
            else None
        )
        failure_reason = simulation_failure or scorer_failure or evaluator_failure
        mask_sample = failure_reason is not None
        reward = assistant_quality if scenario_completed and assistant_quality is not None else 0.0
        return UserSimVerification(
            reward=reward,
            mask_sample=mask_sample,
            failure_kind=(
                "judge_failed"
                if scorer_failure is not None or evaluator_failure is not None
                else ("usersim:simulation_failed" if simulation_failure is not None else None)
            ),
            failure_reason=failure_reason,
            reward_components={
                "participants_completed": float(participants_completed),
                "native_conversation_status": float(native_result.conversation_status),
                "native_scorer_applied": float(native_scorer_name is not None),
                "native_scorer_pass": float(native_scorer_pass),
                "trajectory_evaluator_applied": float(not assistant_eval.get("skipped", False)),
                "assistant_quality": assistant_quality or 0.0,
                **{f"quality.{axis}": score for axis, score in normalized_axis_scores.items()},
            },
            scenario_completed=scenario_completed,
            native_usersim_result=native_result,
            verifier_data={
                "invocations": [invocation.model_dump(mode="json") for invocation in verification_input.invocations],
                "episode_interaction_protocol": verification_input.episode_interaction_protocol,
                "resolved_row": verification_input.resolved_row,
                "usersim_result": native_result.model_dump(mode="json"),
                "native_scorer_name": native_scorer_name,
                "native_scores": native_scores,
                "assistant_eval": assistant_eval,
                "normalized_axis_scores": normalized_axis_scores,
                "scenario_completed": scenario_completed,
            },
        )

    async def close_resources_session(
        self,
        request: Request,
        body: ResourcesCloseSessionRequest,
    ) -> ResourcesCloseSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        seeded = self._seeded_episode(request)
        if body.resources_session_id != seeded.seed.resources_session_id or body.episode_id != seeded.episode_id:
            raise HTTPException(status_code=409, detail="Resources session does not match the active episode")
        lifecycle_task = getattr(seeded.runtime, "_lifecycle_task", None)
        if isinstance(lifecycle_task, asyncio.Task) and not lifecycle_task.done():
            lifecycle_task.cancel()
            await asyncio.gather(lifecycle_task, return_exceptions=True)
        del self.session_id_to_seed[session_id]
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)


def _lifecycle_event(event: Any) -> UserSimLifecycleEvent:
    value = event.to_dict()
    if value.get("complete") is True:
        return UserSimEpisodeLifecycleComplete.model_validate(value)
    return UserSimActivationRequest.model_validate(value)


def _to_responses_input_items(message: Any) -> list[dict[str, Any]]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {"role": getattr(message, "role"), "content": getattr(message, "content", "")}
    role = getattr(value.get("role"), "value", value.get("role"))
    if role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": value["tool_call_id"],
                "output": value.get("content", ""),
            }
        ]
    if role == "assistant" and value.get("tool_calls"):
        items = []
        if value.get("content"):
            items.append({"type": "message", "role": role, "content": value["content"]})
        for call in value["tool_calls"]:
            call = call.model_dump(mode="json", exclude_none=True) if hasattr(call, "model_dump") else dict(call)
            function = call.get("function")
            if isinstance(function, Mapping):
                name = function["name"]
                arguments = function.get("arguments", "{}")
            elif call.get("name"):
                name = call["name"]
                arguments = call.get("arguments_json", call.get("arguments", "{}"))
            else:
                raise ValueError("Assistant tool call must contain a function mapping")
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": name,
                    "arguments": arguments,
                }
            )
        return items
    return [{"type": "message", "role": role, "content": value.get("content", "")}]


def _to_responses_tool(tool: Any) -> dict[str, Any]:
    value = tool.model_dump(mode="json", exclude_none=True) if hasattr(tool, "model_dump") else dict(tool)
    function = value.get("function")
    if isinstance(function, Mapping):
        value = dict(function)
    if not value.get("name"):
        raise ValueError(f"Invalid UserSim function tool schema: {value!r}")
    return {
        "type": "function",
        "name": value["name"],
        "description": value.get("description"),
        "parameters": value.get("parameters", {}),
        "strict": value.get("strict", False),
    }


def _output_message_text(message: NeMoGymResponseOutputMessage) -> str:
    chunks: list[str] = []
    for content in message.content:
        text = getattr(content, "text", None) or getattr(content, "refusal", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _response_text(response: NeMoGymResponse) -> str:
    return "\n".join(
        text
        for item in response.output
        if isinstance(item, NeMoGymResponseOutputMessage)
        if (text := _output_message_text(item))
    )


def _response_reasoning(response: NeMoGymResponse) -> str:
    return "\n".join(
        part.text
        for item in response.output
        if isinstance(item, NeMoGymResponseReasoningItem)
        for part in [*item.summary, *(item.content or [])]
        if getattr(part, "text", None)
    )


def _response_tool_calls(response: NeMoGymResponse) -> list[dict[str, Any]]:
    return [
        {
            "id": item.call_id,
            "type": "function",
            "function": {"name": item.name, "arguments": item.arguments},
        }
        for item in response.output
        if isinstance(item, NeMoGymResponseFunctionToolCall)
    ]


if __name__ == "__main__":
    UserSimResourcesServer.run_webserver()
