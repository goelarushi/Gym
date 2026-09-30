# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for the NeMo UserSim episode protocol."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_resources_server import (
    ResourcesSeedSessionResponse,
    ResourcesVerifyRequest,
)
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle


UserSimAgentRole = Literal["user", "assistant", "judge", "summary"]
USERSIM_EPISODE_PROTOCOL = "usersim.ConversationLoop"


class UserSimScenario(BaseModel):
    """One fully resolved UserSim scenario."""

    model_config = ConfigDict(extra="allow")

    persona: dict[str, Any]
    probe_type: str = "general_open_ended"
    theme: dict[str, Any] | str
    goal: str = ""
    locale: str = "en_US"
    probe_data: dict[str, Any] = Field(default_factory=dict)


class UserSimProtocolConfig(BaseModel):
    """Run-wide ``ConversationSimulatorConfig`` behavior owned by Gym."""

    model_config = ConfigDict(extra="forbid")

    max_tools: int = Field(5, ge=1)
    max_steps: int = Field(10, ge=1)
    max_query_attempts: int = Field(3, ge=1)
    max_assistant_attempts: int = Field(1, ge=1)
    enforce_user_language: bool = True
    user_language_min_script_compliance: float = Field(0.6, ge=0, le=1)
    user_language_min_letters: int = Field(8, ge=0)
    incremental_disclosure_ratio: float = Field(0.6, ge=0, le=1)
    persona_grounding_ratio: float = Field(1, ge=0, le=1)
    context_compression: bool = False
    compression_window: int = Field(1, ge=1)
    store_reasoning: bool = True
    finance_tier_mix: float = Field(0.0, ge=0, le=1)
    finance_tier: Literal["verifiable", "dynamic"] | None = None
    finance_retrieval_mode: Literal["hybrid", "dense", "golden"] = "hybrid"
    finance_embedding_model_alias: str = "embedding_model"
    random_seed: int | None = None
    verbosity: int = Field(1, ge=0, le=2)


class UserSimTaskInput(BaseModel):
    """Fully resolved, provenance-pinned input loaded from one prepared task row."""

    model_config = ConfigDict(extra="forbid")

    scenario: UserSimScenario
    usersim_context: "ResolvedUserSimContext"
    responses_create_params: dict[UserSimAgentRole, NeMoGymResponseCreateParamsNonStreaming] = Field(
        default_factory=dict
    )


class ResolvedUserSimContext(BaseModel):
    """Selection provenance required to replay a resolved scenario."""

    model_config = ConfigDict(extra="forbid")

    locale: str
    seed: int
    personas_dataset_version: str
    personas_panel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    usersim_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class AssistantToolLoopPolicy(BaseModel):
    """Serializable assistant/tool-loop policy supplied by UserSim."""

    model_config = ConfigDict(extra="forbid")

    tool_round_mode: Literal["single", "multi"]
    max_assistant_activations: int = Field(ge=1)
    final_synthesis_without_tools: bool
    single_user_turn: bool
    assistant_error_behavior: Literal["fail_episode"]
    tool_error_behavior: Literal["return_error_payload"]
    max_tool_calls_per_turn: int | None = Field(default=None, ge=1)
    max_tool_response_attempts: int = Field(ge=1)
    assistant_resampling: bool


class UserTurnPolicySnapshot(BaseModel):
    """JSON-safe subset of UserSim's outer user-turn policy."""

    model_config = ConfigDict(extra="forbid")

    context_compression: bool
    wrap_up: bool
    followup_anchor: str | None
    allowed_phrases: list[str]
    script_check_ignores: list[str]
    check_opening: Literal["none", "native"]


class ProbeRuntimeDescriptor(BaseModel):
    """Strict mirror of UserSim's externally hosted runtime descriptor."""

    model_config = ConfigDict(extra="forbid")

    probe_type: str
    assistant_tools: list[dict[str, Any]]
    allowed_tool_names: list[str]
    initial_user_message: str | None
    loop_policy: AssistantToolLoopPolicy
    user_system_prompt: str
    assistant_system_prompt: str
    turn0_user_query_instruction: str | None
    user_interaction_style: str
    patience: float
    user_turn_policy: UserTurnPolicySnapshot


class UserSimSeedResponse(ResourcesSeedSessionResponse):
    """Return resources-session identity plus the resolved scenario."""

    scenario: UserSimScenario
    usersim_context: ResolvedUserSimContext
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
    assistant_tool_loop_policy: AssistantToolLoopPolicy | None = None


class UserSimActivationResult(BaseModel):
    """One externally executed result submitted to the native lifecycle."""

    model_config = ConfigDict(extra="forbid")

    activation_id: str = Field(min_length=1)
    response: dict[str, Any] | None = None
    transcript_delta: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_exactly_one_payload(self) -> "UserSimActivationResult":
        if (self.response is None) == (not self.transcript_delta):
            raise ValueError("exactly one of response or transcript_delta is required")
        return self


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

    scenario: UserSimScenario
    usersim_context: ResolvedUserSimContext
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
