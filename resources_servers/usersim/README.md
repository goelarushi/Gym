# NeMo UserSim Resources Server

This environment initializes one deterministic NeMo UserSim scenario at the
beginning of each `UserSimEnvironmentServer` episode. The Resources Server
loads a persona panel previously created by `usersim panel` during
`gym eval prepare`, hosts episode-scoped simulated tools for agentic probes,
and retains native verification evidence. It does not construct or sample the
panel at runtime.

## Environment initialization

The environment configuration pins the persona dataset version to `0.0.2` and
uses:

```text
environments/usersim/data/personas/
└── 0.0.2/
    └── panels/
        ├── en_US.parquet
        └── en_US.manifest.json
```

For every configured locale, server startup:

1. Requires the panel and manifest prepared by the environment recipe.
2. Validates the panel's version, size, row count, and SHA-256.
3. Loads the panel into memory for prepared-row validation.

Run `gym eval prepare --config environments/usersim/config.yaml` before starting the Resources
Server. Preparation delegates population sampling to NeMo UserSim and treats
the resulting panel as the immutable artifact. Startup fails with that
instruction when the panel is absent or does not match its manifest.

## Episode data contracts

The episode uses separate contracts for each lifecycle. Static protocol and
population settings live in YAML. Preparation expands each tracked template
into a fully resolved task row:

```json
{
  "scenario": {
    "locale": "en_US",
    "persona": {"first_name": "Morgan"},
    "probe_type": "general_open_ended",
    "theme": {"type": "local food", "description": "Seek a practical recommendation."},
    "goal": "Seek a practical recommendation.",
    "probe_data": {}
  },
  "usersim_context": {
    "locale": "en_US",
    "seed": 1042,
    "personas_dataset_version": "0.0.2",
    "personas_panel_sha256": "sha256-without-prefix",
    "usersim_revision": "pinned-40-character-git-revision"
  },
  "responses_create_params": {}
}
```

The Environment Server sends the prepared row to `/seed_session`. The server
returns its executable scenario and immutable provenance after validating them
against the loaded panel:

```json
{
  "scenario": {
    "persona": {"first_name": "Morgan"},
    "probe_type": "general_open_ended",
    "theme": {"type": "local food", "description": "Seek a practical recommendation."},
    "goal": "Seek a practical recommendation.",
    "locale": "en_US"
  },
  "usersim_context": {
    "locale": "en_US",
    "seed": 1042,
    "personas_dataset_version": "0.0.2",
    "personas_panel_sha256": "sha256-without-prefix"
  }
}
```

Scenario content and selection provenance are intentionally separate. The
context does not duplicate the selected persona, probe, theme, or goal.

At `/seed_session`, the server:

1. Validates the row's revision, panel version/checksum, locale, persona, and probe.
2. Stores the resolved context in task-scoped session state.
3. Uses the row's prepared persona, probe, theme, goal, and probe data unchanged.
4. For a tool probe, constructs exactly one `ProbeEpisodeRuntime` and returns its
   serializable descriptor: tool schemas, participant prompts, and tool-loop policy.
5. Returns the resolved scenario to the Environment Server before its first participant
   invocation.

For probes without tools, the Environment Server gives the scenario to NeMo
UserSim's native conversation generator. For tool probes, it drives participant
activations from the Resources-owned runtime descriptor and never constructs a
second probe. It submits each canonical transcript update to that runtime,
then submits the completed episode to `/verify` and closes the Resources
session on every outcome.

`UserSimEnvironmentServer` directly owns this protocol; there is no generic
multi-agent engine. Its native `UserSimEpisodeResponse` contains exactly one
of `result` or `failure`. A successful result retains the verifier output,
native UserSim result, and one ordered `UserSimInvocation` list for User,
Assistant, judge, and summary calls owned by the Environment Server. Each
invocation contains its semantic role, exact Responses API request and response, optional
`AgentObservationBundle`, the environment `state_after` that activation, and
an optional final `termination_reason`. Function calls and
their model-visible results remain ordered inside `response.output`.
Probe API-response calls are Resources Server implementation details and are
retained with probe runtime evidence rather than added to this invocation list.

## Probe tools and episode state

NeMo UserSim selects any Assistant tool schemas required by the resolved probe.
Its runtime also publishes a declarative loop policy, including single- versus
multi-round execution, activation and call limits, and whether a tools-disabled
synthesis pass is required. The Environment Server passes the schemas and
policy only to the Assistant Agent. The Agent owns model-to-tool-to-model
iteration without embedding probe names or probe-specific rules.

Each selected tool is exposed through the standard Resources Server
`POST /{tool_name}` route. Requests include the model's `tool_call_id`, the
semantic Assistant turn index, and call index. The shared Resources session
cookie selects that episode's allowlist, simulated state, and verifier
evidence, so another episode cannot call or mutate those tools. Transcript
synchronization rejects missing, duplicated, reordered, or changed tool
evidence.

The User and Assistant Agents share `policy_model`. The Environment Server
routes UserSim's Judge and Summary calls directly to `support_model`, without
creating support Agent sessions. Resources-owned tool-result synthesis and
native probe scoring retain the purpose-specific `tool_simulation_model` and
`probe_scorer_model` configuration fields, but both reference the same support
Model Server. Policy and support endpoint settings are mandatory and remain
distinct aliases even if explicitly configured to use the same capable
upstream service. `/close_session` removes the resolved scenario and mutable
runtime state.

## Static and dynamic configuration

The YAML config owns static population and probe policy:

- `personas_cache_dir`
- `personas_dataset_version`
- `personas_locales`
- agent, model, and resources-server references
- turn limits
- typed `protocol_config` simulation behavior

Each prepared dataset row owns the complete scenario, selection provenance,
and optional per-role `responses_create_params`. The tracked source templates
live at `data/example_source.jsonl`; preparation writes the ignored runnable
dataset at `environments/usersim/data/example.jsonl`.

Changing the dataset version selects a different prepared-panel cache path.

## Supported probes

`data/example_source.jsonl` contains one tracked template for every first-party
NeMo UserSim probe:

- General: `general_open_ended`, `general_educational`, and `tool_calling`
- Sovereign AI: `sov_ai_facts`, `sov_ai_dynamic`, and
  `sov_ai_multilingual_parity`
- Safety: `safety_chat_pressure` and `safety_agentic`
- Financial services: `financial_services`
- Health disclosure: `health_general_disclosure`,
  `health_therapy_disclosure`, `health_triage_disclosure`, and
  `health_decision_support_disclosure`
- Identity: `identity_disclosure`

The three probes that expose Assistant tools—`tool_calling`,
`safety_agentic`, and `financial_services`—use one episode-scoped external
runtime in the Resources Server. The remaining probes execute their native
UserSim conversation shape through the Environment Server. Asset-backed probes
derive their task from the selected persona and pinned UserSim assets; the
prepared row includes that resolved persona plus a stable locale, seed, probe,
theme, goal, and probe data. The `tool_calling` row additionally supplies its
candidate tool schema.

During `/verify`, the Resources Server invokes UserSim's registered scorer for
tool use, sovereign-AI, safety, financial-services, identity-disclosure, and guarded
health-disclosure trajectories. The four health labels share
`health_disclosure_concealment`; the default health variant has no concealment
ground truth, so that scorer is intentionally not applied. A scorer rejection,
inconclusive status, structured error, or raised exception gates the reward.
Infrastructure, simulated-user, or judge failures also set
`mask_sample=true`; an Assistant failure remains a measured policy outcome and
is not masked.

All completed trajectories also run through UserSim's native trajectory
evaluator. The verifier retains every applicable normalized quality axis and
uses UserSim's `assistant_quality` capability—the mean of normalized
helpfulness, accuracy, and coherence—as the scalar Gym reward. Conversation
completion and any dedicated probe scorer remain prerequisites for receiving
that quality reward. The complete `assistant_eval` and native scorer envelopes
are retained in `verifier_data`.

## Run

Configure the `policy_*` and explicit `support_model_*` endpoint settings,
then prepare the UserSim panel and resolved tasks before collecting rollouts:

```bash
gym eval prepare --config environments/usersim/config.yaml

.venv/bin/gym eval run \
  --environment usersim \
  --split example \
  --output results/usersim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

## Evaluation and training attribution

The native episode result preserves every Environment-owned call under
`invocations`; select either participant policy by semantic role:

```python
selected = [
    {"responses_create_params": call["request"], "response": call["response"]}
    for call in rollout["result"]["invocations"]
    if call["role"] in requested_roles
]
```

Use `{"assistant"}`, `{"user"}`, or both for `requested_roles`. Judge and
Summary support-model calls use distinct roles, while API-response synthesis remains
Resources-owned. Participant filtering provides the explicit per-invocation
contract for downstream SFT, RL projection, or custom collation.
