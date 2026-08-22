"""Cost and off-policy quality for a routing policy.

Two halves, with very different epistemic status, kept deliberately apart:

**Cost is arithmetic.** Given a price sheet, the cost of serving request *x* with
model *m* is a deterministic function of the request's estimated input tokens. No
estimation is involved. The prices themselves are an assumption — the export's
model ids are anonymized, so `scripts/pricing.json` is a stand-in until the
organizers publish one.

**Quality is an estimate, and a weak one.** The log shows only the model that ran,
so the quality of any other choice has to be inferred. The estimator here is
stratified matching: within a complexity stratum, requests served by model *m*
stand in for what *m* would have done on the other requests in that stratum. It
leans on a fact established in the notebook — the logged assignment is close to
complexity-blind (r = +0.03) — which is what makes complexity strata roughly
comparable across models. It cannot rescue a model that served almost nothing.

The observable itself is a proxy: recovered tool-call success rate, not task
success. The export has no final output to score.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PipelineConfig

#: Complexity strata, as quantiles of the training score distribution.
N_STRATA = 4

#: Shrinkage strength for the stratified cell estimate, in pseudo *tool calls*.
#: A cell backed by 25 observed calls is weighted equally against the model's marginal.
SHRINKAGE = 25.0

#: A model must have served at least this many training requests to be a routing
#: candidate. Below it the quality estimate is an anecdote.
DEFAULT_MIN_SUPPORT = 30


def _scripts_on_path(config: PipelineConfig) -> None:
    path = str(config.root / "scripts")
    if path not in sys.path:
        sys.path.insert(0, path)


def load_request_costs(config: PipelineConfig) -> tuple[dict[str, int], dict[str, str], dict]:
    """Estimated input tokens and logged model per request id, plus the price sheet."""
    _scripts_on_path(config)
    from cost_model import load_pricing
    from load_trajectories import est_tokens, iter_requests

    tokens: dict[str, int] = {}
    logged: dict[str, str] = {}
    for chunk, line_no, request in iter_requests(config.export_dir):
        request_id = f"{chunk}:{line_no + 1}"
        tokens[request_id] = est_tokens(request["input"])
        logged[request_id] = request["model"]
    return tokens, logged, load_pricing()


def uncached_rate(model: str, pricing: dict) -> float:
    """Price per estimated input token, uncached."""
    _scripts_on_path(PipelineConfig())
    from cost_model import price_of

    return price_of(model, pricing)[0] / 1e6


def observed_counts(target: dict) -> tuple[float, float]:
    """Tool calls observed and tool calls that errored, for one logged execution.

    Counts, not a ratio. Averaging per-request ratios is what inverted this
    estimator once already: 21% of requests make two or fewer tool calls, so a
    single failure on a 1-call request scores 0.0 and outweighs a 40-call request
    that scored 0.975. Pooling the counts weights every call equally, which is the
    unit the error rate is actually defined over.
    """
    metrics = target.get("observed_metrics") or {}
    outputs = float(metrics.get("tool_output_count", 0) or 0)
    errors = float(metrics.get("tool_error_count", 0) or 0)
    if outputs <= 0:
        return 0.0, 0.0  # nothing observable to score; contributes no weight
    return outputs, min(errors, outputs)


def observed_quality(target: dict) -> float:
    """Per-request success ratio. Kept for inspecting a single request only.

    Do **not** average this across requests to compare models — see
    :func:`observed_counts` for why. :class:`QualityModel` pools counts instead.
    """
    outputs, errors = observed_counts(target)
    if outputs <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - errors / outputs))


@dataclass
class QualityModel:
    """Stratified off-policy estimate of E[tool-call success | model, complexity stratum].

    Fitted on the training split only. Estimates pool *tool calls*, not per-request
    ratios, and are shrunk Beta-Binomial style: a cell's observed successes are
    combined with :data:`SHRINKAGE` pseudo-calls drawn at the model's own marginal
    rate, and that marginal is shrunk the same way toward the global rate. A thin
    cell therefore decays to a broader average instead of to noise.
    """

    edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cell: dict[tuple[str, int], float] = field(default_factory=dict)
    cell_calls: dict[tuple[str, int], float] = field(default_factory=dict)
    by_model: dict[str, float] = field(default_factory=dict)
    model_counts: dict[str, int] = field(default_factory=dict)
    model_calls: dict[str, float] = field(default_factory=dict)
    global_mean: float = 0.5

    def stratum(self, score: float | np.ndarray):
        return np.clip(np.searchsorted(self.edges, score, side="right"), 0, N_STRATA - 1)

    @classmethod
    def fit(cls, scores: np.ndarray, models: list[str], counts: np.ndarray) -> "QualityModel":
        """``counts`` is an (n, 2) array of [tool calls observed, tool calls errored]."""
        counts = np.asarray(counts, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[1] != 2:
            raise ValueError("counts must be an (n, 2) array of [outputs, errors]")

        quantiles = np.linspace(0, 100, N_STRATA + 1)[1:-1]
        edges = np.percentile(scores, quantiles)
        model = cls(edges=edges)

        total_calls = float(counts[:, 0].sum())
        total_errors = float(counts[:, 1].sum())
        model.global_mean = 1.0 - total_errors / total_calls if total_calls > 0 else 0.5

        strata = model.stratum(scores)
        cell_calls: dict[tuple[str, int], float] = defaultdict(float)
        cell_errors: dict[tuple[str, int], float] = defaultdict(float)
        model_calls: dict[str, float] = defaultdict(float)
        model_errors: dict[str, float] = defaultdict(float)
        requests: dict[str, int] = defaultdict(int)

        for name, stratum, (calls, errors) in zip(models, strata, counts):
            requests[name] += 1
            model_calls[name] += calls
            model_errors[name] += errors
            cell_calls[(name, int(stratum))] += calls
            cell_errors[(name, int(stratum))] += errors

        for name in requests:
            calls, errors = model_calls[name], model_errors[name]
            successes = calls - errors
            model.model_counts[name] = requests[name]
            model.model_calls[name] = calls
            model.by_model[name] = (
                (successes + SHRINKAGE * model.global_mean) / (calls + SHRINKAGE)
            )
        for key, calls in cell_calls.items():
            successes = calls - cell_errors[key]
            model.cell_calls[key] = calls
            model.cell[key] = (
                (successes + SHRINKAGE * model.by_model[key[0]]) / (calls + SHRINKAGE)
            )
        return model

    def estimate(self, model: str, stratum: int) -> float:
        key = (model, int(stratum))
        if key in self.cell:
            return self.cell[key]
        return self.by_model.get(model, self.global_mean)

    def support(self, model: str, stratum: int) -> int:
        """Observed tool calls behind this cell — the estimator's real sample size."""
        return int(self.cell_calls.get((model, int(stratum)), 0))

    def candidates(self, min_support: int = DEFAULT_MIN_SUPPORT) -> list[str]:
        """Models with enough logged traffic for their estimate to mean anything."""
        return sorted(m for m, n in self.model_counts.items() if n >= min_support)


def cost_vector(request_ids: list[str], model: str, tokens: dict[str, int],
                pricing: dict) -> np.ndarray:
    rate = uncached_rate(model, pricing)
    return np.array([tokens[r] * rate for r in request_ids], dtype=np.float64)


def cost_matrix(request_ids: list[str], models: list[str], tokens: dict[str, int],
                pricing: dict) -> np.ndarray:
    """(requests x models) cost in USD. Deterministic given the price sheet."""
    token_counts = np.array([tokens[r] for r in request_ids], dtype=np.float64)
    rates = np.array([uncached_rate(m, pricing) for m in models], dtype=np.float64)
    return token_counts[:, None] * rates[None, :]


def quality_matrix(strata: np.ndarray, models: list[str], quality: QualityModel) -> np.ndarray:
    """(requests x models) estimated quality."""
    return np.array(
        [[quality.estimate(m, s) for m in models] for s in strata], dtype=np.float64
    )


def route_tradeoff(
    costs: np.ndarray,
    qualities: np.ndarray,
    lam: float,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pick, per request, the model maximising ``quality - lam * cost``.

    ``lam`` is the exchange rate between estimated quality and dollars. Sweeping it
    from 0 (buy quality at any price) to large (buy the cheapest thing that runs)
    traces the frontier.

    ``candidate_mask`` restricts which columns the router may pick. Every model
    still has a column, so the logged policy and any fixed policy can be costed
    exactly, but a model too thinly observed to estimate is not offered as a choice.
    """
    utility = qualities - lam * costs
    if candidate_mask is not None:
        utility = np.where(candidate_mask[None, :], utility, -np.inf)
    return np.argmax(utility, axis=1)


@dataclass
class PolicyResult:
    name: str
    total_cost: float
    mean_quality: float
    mix: dict[str, int]
    cost_vs_logged: float
    quality_vs_logged: float
    min_cell_support: int
    fallback_share: float


def evaluate_policy(
    name: str,
    choice: np.ndarray,
    models: list[str],
    costs: np.ndarray,
    qualities: np.ndarray,
    strata: np.ndarray,
    quality: QualityModel,
    *,
    logged_cost: float | None = None,
    logged_quality: float | None = None,
    min_support_flag: int = 10,
) -> PolicyResult:
    rows = np.arange(len(choice))
    total_cost = float(costs[rows, choice].sum())
    mean_quality = float(qualities[rows, choice].mean())
    supports = np.array([quality.support(models[c], s) for c, s in zip(choice, strata)])
    mix: dict[str, int] = defaultdict(int)
    for c in choice:
        mix[models[c]] += 1
    return PolicyResult(
        name=name,
        total_cost=total_cost,
        mean_quality=mean_quality,
        mix=dict(sorted(mix.items(), key=lambda kv: -kv[1])),
        cost_vs_logged=(total_cost / logged_cost - 1) if logged_cost else 0.0,
        quality_vs_logged=(mean_quality - logged_quality) if logged_quality is not None else 0.0,
        min_cell_support=int(supports.min()) if len(supports) else 0,
        fallback_share=float((supports < min_support_flag).mean()) if len(supports) else 0.0,
    )


def bootstrap_policy(
    choice: np.ndarray,
    costs: np.ndarray,
    qualities: np.ndarray,
    *,
    draws: int = 500,
    seed: int = 42,
) -> dict:
    """Percentile interval over which held-out requests happened to be sampled.

    This captures sampling error in the evaluation set only. It does **not** capture
    error in the quality estimator itself — for that see
    :func:`bootstrap_quality_estimator`, which refits the estimator on resampled
    training rows and is the wider, honester interval to quote.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(len(choice))
    per_cost = costs[rows, choice]
    per_quality = qualities[rows, choice]
    idx = rng.integers(0, len(choice), size=(draws, len(choice)))
    total_costs = per_cost[idx].sum(axis=1)
    mean_qualities = per_quality[idx].mean(axis=1)
    return {
        "cost_lo": float(np.percentile(total_costs, 2.5)),
        "cost_hi": float(np.percentile(total_costs, 97.5)),
        "quality_lo": float(np.percentile(mean_qualities, 2.5)),
        "quality_hi": float(np.percentile(mean_qualities, 97.5)),
    }


def bootstrap_quality_estimator(
    train_scores: np.ndarray,
    train_models: list[str],
    train_counts: np.ndarray,
    policies: dict[str, list[str]],
    evaluation_scores: np.ndarray,
    *,
    draws: int = 300,
    seed: int = 42,
) -> dict[str, dict]:
    """How much of the quality gap between policies is the estimator guessing?

    Each draw resamples the *training* rows, refits the whole stratified estimator,
    and re-scores every policy against it. The interval that comes back is the one
    that matters when a model's estimate rests on a few dozen logged requests: it
    answers "could this quality difference be an artefact of who happened to serve
    what", which resampling the evaluation set cannot.

    All policies are scored on the same draw, so the paired differences between
    them are meaningful, not just the marginals.
    """
    rng = np.random.default_rng(seed)
    size = len(train_scores)
    samples: dict[str, list[float]] = {name: [] for name in policies}
    paired: dict[str, list[float]] = {name: [] for name in policies}
    reference = policies.get("logged policy")

    for _ in range(draws):
        index = rng.integers(0, size, size)
        refit = QualityModel.fit(
            train_scores[index], [train_models[i] for i in index], train_counts[index]
        )
        strata = refit.stratum(evaluation_scores)
        drawn = {}
        for name, chosen in policies.items():
            drawn[name] = float(np.mean([refit.estimate(m, s) for m, s in zip(chosen, strata)]))
            samples[name].append(drawn[name])
        if reference is not None:
            base = drawn["logged policy"]
            for name in policies:
                paired[name].append(drawn[name] - base)

    out = {}
    for name in policies:
        values = np.array(samples[name])
        row = {
            "quality_mean": float(values.mean()),
            "quality_lo": float(np.percentile(values, 2.5)),
            "quality_hi": float(np.percentile(values, 97.5)),
        }
        if reference is not None:
            diffs = np.array(paired[name])
            row.update(
                delta_vs_logged_mean=float(diffs.mean()),
                delta_vs_logged_lo=float(np.percentile(diffs, 2.5)),
                delta_vs_logged_hi=float(np.percentile(diffs, 97.5)),
                # A sign that survives resampling is the only kind worth quoting.
                delta_excludes_zero=bool(
                    np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0
                ),
            )
        out[name] = row
    return out
