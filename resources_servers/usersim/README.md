# NeMo UserSim Resources Server

The Resources Server hosts one canonical UserSim runtime per Gym episode. The
prepared task contains an unchanged `resolved_row` produced by UserSim at
revision `a4665b3ce1a030e83871232e2fb69e5b39480818`.

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

The Assistant Agent retains the mechanical model → tool → model loop, but it
does not decide the loop's shape. Per step it:

1. calls its model;
2. records that response with `POST /runtime/record`, **before** issuing the
   response's tool calls;
3. reads the reply — whether the turn continues, which tools are offered next,
   and the exact input UserSim would send;
4. collects each executed call's payload from `POST /{tool_name}`, identified
   only by `X-NeMo-Gym-Tool-Call-Id`.

Recording is what makes UserSim execute the calls, inside its own conversation
loop and with its own turn and call indices. Gym sends no turn or call index
headers, and there is no batch route: parallel calls arrive together in one
recorded response.

The reply's input replaces the Agent's own accumulated transcript. The two
diverge for probes that trim their context, and UserSim's is the one the
episode is scored on.

Payloads are returned as opaque plain text, without JSON parsing or a wrapper;
several probes' simulated responses are not JSON. A probe caps how many calls
it runs per turn, so `executed_tool_call_ids` on the reply is the authoritative
list of which recorded calls actually ran.

A recorded response that breaks the probe's loop rules — tool calls when the
activation offers none, an unoffered tool name, a missing or repeated call id —
returns HTTP 422. These are host bugs, so they never consume a model retry and
are never attributed to the model under test.

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
