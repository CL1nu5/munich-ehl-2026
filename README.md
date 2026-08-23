# Viktor learned model router

Offline, dependency-free router for the TUM.ai Viktor Challenge.

## Final pipeline

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

The raw export is never modified.

## Pitch demo

Replay two real held-out decisions—one cost-saving downgrade and one quality-
guardrail up-route—with no API calls:

```bash
python scripts/demo_router.py --pause
```

The demo reads the untouched test-set result table and is designed to take
about 35 seconds during the five-minute pitch.

## Architecture

1. **Workload model:** predicts a 0–100 execution-workload score from only the
   opening context. Its target combines later token growth, tool calls, and
   distinct tools. Numeric features are augmented with sanitized TF-IDF features
   from the opening instruction.
2. **Execution-risk model:** separately predicts whether the run will contain a
   tool error, user correction, or duplicate successful side effect. Failures are
   not treated as intrinsic task workload.
3. **Quality model:** predicts task difficulty and adds a cross-validated,
   non-negative benchmark-capability effect for each candidate. Model price is
   deliberately excluded from quality prediction and used only by the policy.
4. **Learned policy:** among candidates with historical feature overlap, choose
   the cheapest model within the validation-selected quality tolerance of the
   best predicted model. Requests outside all support regions use a fixed
   Sonnet fallback; the logged model is never required at deployment.

The train/validation/test split keeps recurring sanitized task templates intact,
so near-identical scheduled jobs cannot cross partitions. Stage-one inputs are
generated with template-grouped folds for quality-model training. Final quality
is reported with a stabilized doubly robust off-policy estimate on a
template-held-out test set. Policy selection also requires minimum overlap ESS
and agreement with the direct outcome model.

The selected `epsilon = 0.025` policy saves **47.6%** of assumed opening-input
cost on the 198-row template-held-out test set. Its stabilized doubly robust
quality delta is **+0.018**, with bootstrap 95% CI **[-0.008, +0.042]**. The
defensible claim is lower assumed cost with no detected quality loss, not a
proven quality increase.

## Important artifacts

| Path | Purpose |
|---|---|
| `models/benchmark_priors.json` | Assumed public-analogue capability priors and source links |
| `models/two_stage_router.json` | Final model refit on all usable rows |
| `results/complexity_scores.csv` | Observed and predicted complexity per usable trajectory |
| `results/two_stage_validation_frontier.csv` | Validation policy-selection frontier |
| `results/two_stage_test.csv` | Template-held-out test decisions and estimates |
| `results/two_stage_summary.md` | Main result, methodology, and limitations |
| `results/two_stage_routes.jsonl` | Final route for every corrected trajectory |
| `results/evaluator_comparison.csv` | Same-test incumbent, heuristic, and learned frontier |
| `results/benchmark_sensitivity.csv` | Alternative benchmark-prior mappings |
| `results/ope_weight_sensitivity.csv` | Clipped importance-weight robustness diagnostic |
| `results/cost_quality_frontier.svg` | Presentation-ready final frontier chart |
| `results/evaluator_summary.md` | Final evaluator claim and limitations |

## Analysis notebooks

The notebooks are pre-rendered, so they open with their charts and findings even
without rerunning a kernel. Rebuild them and the standalone figures with
`python scripts/build_analysis_notebooks.py`.

| Path | Purpose |
|---|---|
| `notebooks/01_data_and_label_audit.ipynb` | Data mix, logged-model coverage, and outcome-label balance |
| `notebooks/02_complexity_and_quality_diagnostics.ipynb` | Complexity accuracy, category effects, predictive uncertainty, and quality calibration |
| `notebooks/03_router_frontier_and_ope.ipynb` | Held-out frontier, overlap diagnostics, and sensitivity analyses |
| `results/figures/` | SVG and PNG versions of every notebook chart for presentations |

## Assumptions

- Token counts are estimated as serialized characters / 4; no measured usage is
  present in the export.
- The challenge IDs are anonymized. Capability priors are normalized ordinal
  public-analogue assumptions, not verified mappings or copied leaderboard
  percentages; cross-family spacing is tested in sensitivity analysis.
- Default prices are a public-analogue scenario checked on 2026-08-22, not the
  challenge's factual prices. Replace them with `scripts/pricing.json` if the
  organizers provide a price sheet.
- Only the factual model outcome is observed. Counterfactual quality remains an
  off-policy estimate, not randomized evidence.
