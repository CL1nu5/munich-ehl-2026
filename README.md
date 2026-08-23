# Viktor model router

An offline, dependency-free model router for the TUM.ai Viktor Challenge. It
reconstructs redacted request trajectories, learns a cache-aware routing policy,
and evaluates that policy with explicit off-policy assumptions.

## What it does

Given the opening context of a request, the router estimates execution workload,
execution risk, and expected outcome quality for supported model choices. It then
selects the least expensive supported model within a validation-selected quality
tolerance of the best predicted option.

The pipeline is deliberately conservative:

- Routing inputs use only information available before execution.
- Collision-prone trajectory candidates are audited; ambiguous candidates stop
  for manual review rather than being silently merged.
- The logged model is treated as the factual treatment, never as a deployment
  input.
- Quality comparisons use a stabilized doubly robust off-policy estimate and
  report overlap diagnostics.

## Quick start

Use Python 3. The project uses only the standard library.

1. Extract the supplied archives into `export/` so it contains
   `trajectories_v1_*.jsonl` files.
2. Run the pipeline from the repository root:

```bash
python scripts/load_trajectories.py export/
python scripts/audit_trajectory_groups.py export/
python scripts/build_feature_table.py export/
python scripts/build_evaluation_table.py export/
python scripts/build_manual_review_sample.py export/
python scripts/train_two_stage_router.py
python scripts/two_stage_router.py
python scripts/evaluate_two_stage_router.py
python scripts/build_analysis_notebooks.py
```

Generated tables, charts, and local evaluation artifacts are written to
`results/`. They are intentionally ignored by Git.

For a short offline replay of two held-out decisions:

```bash
python scripts/demo_router.py --pause
```

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/load_trajectories.py` | Parses the export and reconstructs candidate trajectories from opening system and user context. |
| `scripts/audit_trajectory_groups.py` | Splits demonstrable opening-context collisions and halts for ambiguous candidates. |
| `scripts/build_feature_table.py` | Builds leakage-aware opening-context routing features. |
| `scripts/build_evaluation_table.py` | Derives observable outcome proxies from recoverable history. |
| `scripts/build_manual_review_sample.py` | Applies the fixed, chunk-scoped manual calibration sample. |
| `scripts/train_two_stage_router.py` | Trains the workload, risk, quality, and routing policy models. |
| `scripts/two_stage_router.py` | Produces a route for every corrected request. |
| `scripts/evaluate_two_stage_router.py` | Builds the evaluator comparison, sensitivity tables, and frontier chart. |
| `models/` | The router artifact and documented benchmark-capability priors. |
| `notebooks/` | Pre-rendered analysis notebooks. |

## Evaluation and limits

The export contains no final responses, measured usage, timing, or randomized
counterfactuals. Consequently:

- Token counts are estimates based on serialized characters divided by four.
- Cache savings are inferred from item-level shared input prefixes; they are not
  provider-reported cached-token measurements.
- Prices and cross-family capability priors are explicit public-analogue
  assumptions. Replace them if organizers provide a price sheet.
- Observable tool history is a quality proxy, not a semantic ground-truth label.
- A quality delta is an off-policy estimate, not proof that a new route improves
  outcomes.

Read the generated `results/two_stage_summary.md` and
`results/evaluator_summary.md` alongside the frontier rather than quoting a
single point estimate.

## Data handling

`export/` is challenge-use-only data. It is ignored by Git and must not be
committed, uploaded, or redistributed. The pipeline reads it but never modifies
it.
