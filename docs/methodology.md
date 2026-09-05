# Method and limitations

The policy makes one decision at the start of a task and keeps that model for
the trajectory. Its objective is to reduce assumed input cost while staying
close to the best predicted quality among historically supported candidates.

## What the export actually contains

Each JSONL record has a model ID, the request history, and tool definitions.
There are no trajectory IDs, final responses, measured token usage, or quality
labels. Model IDs are anonymized.

`load_trajectories.py` groups requests by the first system message and the first
2,000 characters of the first user message, then orders them by input length.
This produces candidates, not guaranteed trajectories. The audit splits
demonstrable collisions and stops on unresolved groups. Its current rules are
specific to the challenge export; even a genuine growing-history chain requires
review rather than being automatically accepted by that audit.

Earlier assistant messages and tool calls can appear in later request histories.
Those histories supply workload and outcome signals. The final response remains
unobserved. Outcome labels primarily reflect tool success, failures, and
corrections, rather than whether the final answer solved the user's task.

The calibration script also applies a fixed 100-case observable-history review
recorded during the hackathon. Those judgments were made with Codex and are
scoped to chunk 01 and its line numbers. They are not independent human labels
or a reusable judge for new exports.

## Routing

The feature table separates opening-time inputs from later execution signals.
Task text is sanitized before text features are extracted. The logged model is
an observed treatment for evaluation; it is not an input to the routing decision.

Two ridge-regression heads estimate workload and execution risk. The workload
target combines continuation token estimates, tool-call count, and distinct
tools. These predictions feed a separate quality model alongside task features
and assumed model-capability profiles. Price enters the final selection, not
the quality prediction.

For each candidate, support requires enough matching historical examples in the
task-category/risk/image stratum and a nearby opening-feature vector. Of the supported
candidates, the router keeps those within `epsilon` of the highest predicted
quality and chooses the cheapest. If none has support, it uses a fixed Sonnet
fallback. That fallback is a routing rule, not evidence of a reliable
counterfactual estimate for that request.

The saved artifact uses `epsilon = 0.025` and a minimum exact-stratum size of 3.
It was fitted on 991 usable rows after policy selection. Its predictions on
arbitrary new requests are not additional held-out results.

## Evaluation

Task templates are kept together in a deterministic train/validation/test split.
Stage-one predictions for training the quality model are generated out of fold.
Validation selects the quality tolerance subject to quality, savings, and
overlap checks. The test frontier is diagnostic; it does not select the final
tolerance.

The evaluator compares the logged route, a 15k-token heuristic, and learned
policies on the same test rows. A stabilized doubly robust estimator combines
quality predictions with propensity-weighted corrections where the policy's
choice matches the logged model. Bootstrap intervals and effective sample size
describe uncertainty and overlap. Additional runs vary importance-weight caps
and model-capability assumptions.

This still depends on measured features adequately explaining model assignment
and outcome differences. Unrecorded task difficulty or assignment rules can bias
the estimate. Stabilization and bootstrapping do not remove that bias, and the
intervals do not capture every pricing, labeling, or reconstruction uncertainty.

## Cost and caching

All token counts use a characters-divided-by-four estimate. The numbers in
`scripts/cost_model.py` and the capability profiles in
`models/benchmark_priors.json` are the hackathon's assumed analogue scenario.
The anonymized IDs do not establish a correspondence to publicly sold models;
the included source URLs are historical references, not verified mappings or
current pricing evidence.

There are two distinct cost calculations:

| Calculation | Scope |
| :--- | :--- |
| Router frontier | Opening input, including tool schemas, at the assumed uncached input rate |
| `trajectory_cost()` | Request-history input across calls, with shared item prefixes priced as cached when the model stays the same |

The trajectory helper resets the inferred cache on a model switch. It does not
include tool-schema cost or output-token cost. Although some earlier outputs
can be recovered from later histories, this implementation does not bill them
as output tokens. Neither calculation is a measured end-to-end provider bill.

An organizer price sheet can override the cost model via `scripts/pricing.json`.
Rerun training and evaluation when changing prices: the saved router carries
its own pricing table.

## Reading the saved result

The committed notebooks describe a 1,000-request run with 991 usable labels and
198 test trajectories. The selected policy reports:

| Quantity | Saved value |
| :--- | ---: |
| Assumed opening-input savings | 47.6% |
| Estimated quality-proxy change | +0.0179 |
| 95% bootstrap interval for the change | [−0.0075, +0.0422] |
| Effective overlap sample size | 54.2 |

The interval crosses zero. The result suggests a useful cost trade-off under
the stated assumptions, while leaving a small quality loss plausible. Missing
final answers and unobserved counterfactual outcomes are the main limits.

The README frontier is extracted from the saved output of
[`03_router_frontier_and_ope.ipynb`](../notebooks/03_router_frontier_and_ope.ipynb).
Its labels and contrast have been tidied up; plot coordinates are unchanged.
It is an archived figure, independent of subsequent local runs.
