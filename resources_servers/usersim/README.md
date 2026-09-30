# NeMo UserSim Resources Server

The Resources Server hosts one canonical UserSim runtime per Gym episode. The
prepared task contains an unchanged `resolved_row` produced by UserSim at
revision `4fd4c800bbef8883329543df632f328860fc6429`.

At `/seed_session`, Resources validates the row's UserSim revision and
trajectory identity, then constructs the episode only through:

```python
ProbeEpisodeRuntime.from_resolved_row(resolved_row, models=runtime_models)
```

Gym does not rebuild persona behavior, themes, toolsets, configuration,
provenance, or trajectory IDs. UserSim remains authoritative for those values.
The seed response returns the same resolved row plus UserSim's runtime
descriptor and Assistant tool schemas.

## Tool execution

The Assistant Agent retains the mechanical model → tool → model loop.
Resources exposes:

- `POST /{tool_name}` for one call, identified only by
  `X-NeMo-Gym-Tool-Call-Id`.
- `POST /runtime/tool_calls` for an ordered parallel batch.

UserSim assigns semantic turn and call indices. Gym does not send host turn or
call index headers. Single-call responses return UserSim's payload as opaque
plain text without JSON parsing or a wrapper. Batch responses preserve each
payload string in request order.

The shared Resources session cookie selects the episode allowlist, state, and
evidence. Identical UserSim call-ID retries are idempotent; conflicting reuse
is rejected by UserSim.

## Lifecycle and verification

The Environment Server drives `/runtime/start` and `/runtime/advance`, routing
UserSim's typed User and Assistant activations to Agents and Judge/Summary
activations to the support model. Resources owns runtime state, native scoring,
and finalization.

The User and Assistant Agents share `policy_model`. Tool-response synthesis,
native scoring, Judge, and Summary use the separately configured
`support_model`. Successful episode output retains the unchanged resolved row,
native UserSim result, verifier evidence, and ordered participant invocations.

## Run

```bash
gym eval prepare --config environments/usersim/config.yaml

gym eval run \
  --environment usersim \
  --split benchmark \
  --output results/usersim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

Preparation uses UserSim's canonical sampler to generate one resolved row for
every registered probe. The Resources-owned dataset config is the single
source of Gym dataset routing.
