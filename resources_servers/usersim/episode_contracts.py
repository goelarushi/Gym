# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the NeMo UserSim episode protocol."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.base_resources_server import (
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesVerifyRequest,
)
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


UserSimAgentRole = Literal["user", "assistant", "judge", "summary"]
USERSIM_EPISODE_PROTOCOL = "usersim.ConversationLoop"


class UserSimTaskInput(BaseModel):
    """Fully resolved, provenance-pinned input loaded from one prepared task row."""

    model_config = ConfigDict(extra="forbid")

    resolved_row: dict[str, Any]
    responses_create_params: dict[UserSimAgentRole, NeMoGymResponseCreateParamsNonStreaming] = Field(
        default_factory=dict
    )


class ProbeRuntimeDescriptor(BaseModel):
    """Strict mirror of UserSim's externally hosted runtime descriptor.

    Deliberately small. Every per-call decision -- whether tools are offered,
    whether the turn continues, what to send next -- rides on the activation
    reply, so neither this environment nor the Agent models UserSim's loop.
    """

    model_config = ConfigDict(extra="forbid")

    probe_type: str
    assistant_tools: list[dict[str, Any]] = Field(default_factory=list)


class UserSimRoleModel(BaseModel):
    """The model identity and token budget this environment runs for one role.

    UserSim keys trajectory identity on the resolved model id, and
    ``identity_disclosure`` cannot grade an assistant it cannot name, so the
    host declares what is behind each role rather than letting UserSim fall
    back to the models preparation happened to be configured with.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)


class UserSimSeedSessionRequest(ResourcesSeedSessionRequest):
    """Seed one episode, declaring the models this environment will run."""

    role_models: dict[str, UserSimRoleModel] = Field(default_factory=dict)


class UserSimSeedResponse(ResourcesSeedSessionResponse):
    """Return resources-session identity plus the unchanged resolved row."""

    resolved_row: dict[str, Any]
    assistant_tools: list[dict[str, Any]] = Field(default_factory=list)
    runtime_descriptor: ProbeRuntimeDescriptor | None = None


class UserSimActivationRequest(BaseModel):
    """One model activation requested by the Resources-owned native lifecycle."""

    model_config = ConfigDict(extra="forbid")

    activation_id: str = Field(min_length=1)
    role: UserSimAgentRole
    model_alias: str = Field(min_length=1)
    messages: list[dict[str, Any]]
    parameters: dict[str, Any]
    #: Tools UserSim offers for this call. Empty means tools are off, which is
    #: how a probe asks the assistant for its final answer.
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tools_enabled: bool = False
    #: True when this activation continues the turn already in progress. The
    #: Assistant Agent keeps its loop running while this stays true.
    continues_turn: bool = False


class UserSimActivationUsage(BaseModel):
    """Token usage observed for one activation."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class UserSimActivationResult(BaseModel):
    """One model response recorded against the native lifecycle.

    The response is recorded, tool calls included, before those calls are
    issued; UserSim's own loop executes them as part of recording.
    """

    model_config = ConfigDict(extra="forbid")

    activation_id: str = Field(min_length=1)
    response: dict[str, Any]
    usage: UserSimActivationUsage | None = None


class UserSimSimulationResult(BaseModel):
    """Typed view of UserSim's Data Designer-compatible output columns."""

    model_config = ConfigDict(extra="allow")

    conversation_messages: list[dict[str, Any]]
    conversation_status: bool
    simulation_outcome: dict[str, Any] = Field(default_factory=dict)
    conversation_metadata: dict[str, Any] | None = None
    simulation_traces: list[dict[str, Any]] | None = None

    @field_validator(
        "conversation_messages",
        "simulation_outcome",
        "conversation_metadata",
        "simulation_traces",
        mode="before",
    )
    @classmethod
    def decode_json_columns(cls, value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value


class UserSimEpisodeLifecycleComplete(BaseModel):
    """Terminal event from the Resources-owned native lifecycle."""

    model_config = ConfigDict(extra="forbid")

    complete: Literal[True] = True
    result: UserSimSimulationResult


UserSimLifecycleEvent = UserSimActivationRequest | UserSimEpisodeLifecycleComplete


class UserSimAgentRecordRequest(BaseModel):
    """One Assistant-Agent model response recorded against the live episode.

    No activation id: the session knows which activation it is waiting on, so
    the Agent records a model response and nothing else.
    """

    model_config = ConfigDict(extra="forbid")

    response: dict[str, Any]
    usage: UserSimActivationUsage | None = None


class UserSimAgentRecordResponse(BaseModel):
    """What UserSim tells the Assistant Agent to do next.

    The Agent owns its model/tool loop, but it does not decide the loop's
    shape: it records each response, then follows this reply. ``input`` is the
    exact input UserSim would send next, which is why the Agent replaces its
    own accumulated transcript with it -- the two diverge for probes that trim
    their context.
    """

    model_config = ConfigDict(extra="forbid")

    #: True while the next activation continues this assistant turn.
    should_continue: bool
    #: The next activation's identity, when the turn continues.
    activation_id: str | None = None
    #: The input UserSim would send next, already as Responses input items so
    #: a general-purpose Agent never has to know UserSim's message shape.
    input: list[dict[str, Any]] = Field(default_factory=list)
    #: Tools offered for the next call, as Responses function tools. Empty
    #: means the probe turned them off for that call.
    tools: list[dict[str, Any]] = Field(default_factory=list)
    #: Tool payloads UserSim's loop produced for the recorded call ids, in
    #: execution order. A probe may cap how many calls it runs per turn, so a
    #: recorded call can be absent.
    executed_tool_call_ids: list[str] = Field(default_factory=list)
    #: True once the episode has finished; the Agent stops either way.
    complete: bool = False


class UserSimInvocation(BaseModel):
    """One ordered UserSim participant-Agent or support-model activation."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    role: UserSimAgentRole
    request: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    observations: AgentObservationBundle | None = None
    state_after: dict[str, Any] | None = None
    termination_reason: str | None = None


class UserSimVerificationInput(BaseModel):
    """Carry the completed protocol to the Resources Server verifier."""

    model_config = ConfigDict(extra="forbid")

    resolved_row: dict[str, Any]
    usersim_result: UserSimSimulationResult
    invocations: list[UserSimInvocation]
    episode_interaction_protocol: str = USERSIM_EPISODE_PROTOCOL


class UserSimVerifyRequest(ResourcesVerifyRequest[UserSimVerificationInput]):
    """Verify one completed UserSim episode."""


class UserSimVerification(BaseModel):
    """Typed verifier output preserved in the Environment Server result."""

    model_config = ConfigDict(extra="forbid")

    reward: float
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None
    reward_components: dict[str, float]
    scenario_completed: bool
    verifier_data: dict[str, Any] = Field(default_factory=dict)
    native_usersim_result: UserSimSimulationResult | None = None


class UserSimEpisodeResult(BaseModel):
    """Successful UserSim episode output."""

    model_config = ConfigDict(extra="forbid")

    verification: UserSimVerification
    usersim_result: UserSimSimulationResult
    invocations: list[UserSimInvocation]
    episode_interaction_protocol: str = USERSIM_EPISODE_PROTOCOL


class UserSimEpisodeFailure(EpisodeFailure):
    """Add the failing UserSim protocol stage."""

    stage: Literal["seed", "participant", "simulation", "verification", "cleanup"] | None = None


class UserSimEpisodeRequest(BaseEpisodeRequest[UserSimTaskInput]):
    """Native request for the UserSim episode protocol."""


class UserSimEpisodeResponse(BaseEpisodeResponse[UserSimEpisodeResult]):
    """Native response for the UserSim episode protocol."""

    failure: UserSimEpisodeFailure | None = None
