"""The cost-quality frontier: what routing actually saves, and what it costs in quality.

The evaluation makes one distinction that decides whether the complexity model
matters at all:

* the router **chooses** using the *predicted* complexity, because that is all it
  has at routing time;
* the choice is **scored** against the request's *true* complexity stratum.

So a request the model thinks is easy but that is really hard gets a weak model and
is charged the weak model's estimated quality *on hard requests*. Prediction error
shows up as lost quality rather than being quietly forgiven.

Everything the estimator needs is fitted on the training split; the frontier is
reported on the held-out test split.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .config import PipelineConfig
from .routing import (
    DEFAULT_MIN_SUPPORT,
    QualityModel,
    bootstrap_policy,
    bootstrap_quality_estimator,
    cost_matrix,
    evaluate_policy,
    load_request_costs,
    observed_counts,
    quality_matrix,
    route_tradeoff,
    uncached_rate,
)

#: Exchange rates between estimated quality and dollars, swept to trace the curve.
LAMBDA_GRID = np.concatenate([[0.0], np.logspace(-2, 4, 60)])

CHEAP_SIBLING = {"claude": "claude-sonnet-5", "gpt": "gpt-5.6-luna"}
SMALL_TRAJECTORY_TOKENS = 15_000


def _baseline_choice(request_ids, tokens, logged, models) -> np.ndarray:
    """`scripts/baseline_router.py`: small requests go to a cheaper family sibling."""
    index = {m: i for i, m in enumerate(models)}
    out = []
    for request_id in request_ids:
        current = logged[request_id]
        if tokens[request_id] < SMALL_TRAJECTORY_TOKENS:
            family = "claude" if current.startswith("claude") else "gpt"
            candidate = CHEAP_SIBLING[family]
            current = candidate if candidate in index else current
        out.append(index.get(current, index[logged[request_id]]))
    return np.array(out, dtype=int)


def pareto_front(points: list[dict], *, tolerance: int = 4) -> list[dict]:
    """Keep points that no other point beats on both cost and quality.

    The lambda sweep piles many settings onto the same routing decision once the
    cost term dominates, so identical (cost, quality) outcomes are collapsed to one
    representative rather than repeated down the frontier.
    """
    front = []
    seen: set[tuple[float, float]] = set()
    for point in points:
        signature = (round(point["total_cost"], tolerance), round(point["mean_quality"], tolerance))
        if signature in seen:
            continue
        dominated = any(
            other["total_cost"] <= point["total_cost"]
            and other["mean_quality"] >= point["mean_quality"]
            and (other["total_cost"] < point["total_cost"]
                 or other["mean_quality"] > point["mean_quality"])
            for other in points
        )
        if not dominated:
            seen.add(signature)
            front.append(point)
    return sorted(front, key=lambda p: p["total_cost"])


def build_frontier(
    report: dict,
    config: PipelineConfig | None = None,
    *,
    split: str = "test",
    min_support: int = DEFAULT_MIN_SUPPORT,
    verbose: bool = True,
) -> dict:
    """Cost and estimated quality for the logged policy, fixed policies, and the sweep."""
    config = config or PipelineConfig(**{})
    splits = report["_splits"]
    head = report["_head"]
    features = report["_features"]

    tokens, logged_model, pricing = load_request_costs(config)

    # --- fit the off-policy estimator on training rows only -----------------
    train = splits["train"]
    train_scores = np.array([row["complexity_score"] for row in train.targets])
    train_counts = np.array([observed_counts(row) for row in train.targets])
    train_models = [row["logged_model"] for row in train.metadata]
    quality = QualityModel.fit(train_scores, train_models, train_counts)

    # Every logged model gets a column so any policy can be costed exactly; only
    # well-supported models are offered to the router as choices.
    models = sorted(quality.model_counts)
    candidates = quality.candidates(min_support)
    candidate_mask = np.array([m in set(candidates) for m in models])
    if verbose:
        dropped = sorted(set(models) - set(candidates))
        print(f"routing candidates ({len(candidates)}, >={min_support} training requests): "
              f"{', '.join(candidates)}")
        if dropped:
            print(f"excluded for thin support: "
                  f"{', '.join(f'{m} (n={quality.model_counts[m]})' for m in dropped)}")

    # --- held-out split ------------------------------------------------------
    evaluation = splits[split]
    request_ids = evaluation.request_ids
    true_scores = np.array([row["complexity_score"] for row in evaluation.targets])
    predicted_scores = head.predict(features[split])[:, 0]

    true_strata = quality.stratum(true_scores)
    predicted_strata = quality.stratum(predicted_scores)

    costs = cost_matrix(request_ids, models, tokens, pricing)
    routing_quality = quality_matrix(predicted_strata, models, quality)  # what the router sees
    scoring_quality = quality_matrix(true_strata, models, quality)       # what it is scored on

    index = {m: i for i, m in enumerate(models)}

    # --- reference policies --------------------------------------------------
    logged_choice = np.array([index[logged_model[r]] for r in request_ids], dtype=int)
    rows = np.arange(len(request_ids))
    logged_cost = float(costs[rows, logged_choice].sum())
    logged_quality_value = float(scoring_quality[rows, logged_choice].mean())

    results = []

    def record(name, choice, *, kind):
        result = evaluate_policy(
            name, choice, models, costs, scoring_quality, true_strata, quality,
            logged_cost=logged_cost, logged_quality=logged_quality_value,
        )
        row = asdict(result)
        row["kind"] = kind
        row["_choice"] = choice
        row.update(bootstrap_policy(choice, costs, scoring_quality))
        results.append(row)
        return row

    logged_row = record("logged policy", logged_choice, kind="reference")
    record("baseline heuristic", _baseline_choice(request_ids, tokens, logged_model, models),
           kind="reference")
    for name in models:
        record(f"always {name}", np.full(len(request_ids), index[name], dtype=int),
               kind="single model")
        results[-1]["training_support"] = quality.model_counts[name]
        results[-1]["is_candidate"] = bool(name in set(candidates))

    # --- the sweep -----------------------------------------------------------
    sweep = []
    for lam in LAMBDA_GRID:
        choice = route_tradeoff(costs, routing_quality, float(lam), candidate_mask)
        row = record(f"router lambda={lam:.4g}", choice, kind="router")
        row["lambda"] = float(lam)
        sweep.append(row)

    front = pareto_front([
        r for r in results
        if r["kind"] == "router"
        or (r["kind"] == "single model" and r.get("is_candidate"))
    ])

    # --- how much of the quality gap is the estimator guessing? --------------
    def chosen_models(row):
        return [models[c] for c in row["_choice"]]

    key_policies = {"logged policy": chosen_models(logged_row)}
    for row in results:
        if row["kind"] == "reference" and row["name"] != "logged policy":
            key_policies[row["name"]] = chosen_models(row)
    for row in front:
        key_policies[row["name"]] = chosen_models(row)
    uncertainty = bootstrap_quality_estimator(
        train_scores, train_models, train_counts, key_policies, true_scores,
    )
    for row in results:
        if row["name"] in uncertainty:
            row["uncertainty"] = uncertainty[row["name"]]
    for row in results:
        row.pop("_choice", None)

    payload = {
        "split": split,
        "n_requests": len(request_ids),
        "pricing_is_an_assumption": True,
        "models": models,
        "candidates": candidates,
        "min_support": min_support,
        "excluded_models": {m: quality.model_counts[m]
                            for m in sorted(set(models) - set(candidates))},
        "logged": logged_row,
        "policies": results,
        "sweep": sweep,
        "pareto": front,
        "quality_uncertainty": uncertainty,
        "quality_support": {
            "global_mean": quality.global_mean,
            "by_model": {m: {"quality": round(quality.by_model[m], 4),
                             "n": quality.model_counts[m],
                             "tool_calls": int(quality.model_calls[m])}
                         for m in sorted(quality.model_counts)},
            "stratum_edges": [round(float(e), 2) for e in quality.edges],
        },
    }

    if verbose:
        best = min((r for r in results if r["kind"] == "router"),
                   key=lambda r: r["total_cost"] - 0 * r["mean_quality"])
        print(f"\nlogged policy on {split}: ${logged_cost:,.2f}, "
              f"estimated quality {logged_quality_value:.4f}")
        print(f"cheapest router point:     ${best['total_cost']:,.2f} "
              f"({best['cost_vs_logged']:+.1%}), quality {best['mean_quality']:.4f} "
              f"({best['quality_vs_logged']:+.4f})")
    return payload


def frontier_table(payload: dict, kinds=("reference", "single model")) -> list[dict]:
    rows = []
    for row in payload["policies"]:
        if row["kind"] not in kinds:
            continue
        rows.append({
            "policy": row["name"],
            "cost_usd": round(row["total_cost"], 2),
            "vs_logged": f"{row['cost_vs_logged']:+.1%}",
            "est_quality": round(row["mean_quality"], 4),
            "quality_delta": f"{row['quality_vs_logged']:+.4f}",
            "min_support": row["min_cell_support"],
        })
    return sorted(rows, key=lambda r: r["cost_usd"])


#: Categorical slots 1-3 of the reference palette. This is a scatter, so the
#: all-pairs pairlist applies; the documented validation for these three slots is
#: CVD dE 9.2 / normal-vision dE 24.0 on the light surface, clear of both floors.
PALETTE = {
    "router": "#2a78d6",     # slot 1, blue
    "single": "#eb6834",     # slot 2, orange
    "baseline": "#1baf7a",   # slot 3, aqua
    "ink": "#0b0b0b",
    "ink_soft": "#52514e",
    "surface": "#fcfcfb",
    "grid": "#dededa",
}


def plot_frontier(payload: dict, path=None):
    """Cost against estimated quality, with a ranked policy table beside it.

    The table carries the identities so the plot needs only a few direct labels —
    scattering one annotation per model is what made the earlier version unreadable.
    It doubles as the required non-visual view of the same numbers.

    The shaded band is the 95% interval from refitting the quality estimator on
    resampled training rows. Where it swallows the vertical gap to the logged
    marker, the frontier is not claiming a difference it can defend.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from pathlib import Path

    uncertainty = payload.get("quality_uncertainty", {})
    logged = payload["logged"]
    sweep = payload["sweep"]
    front = payload["pareto"]
    candidates = set(payload["candidates"])
    singles = sorted((r for r in payload["policies"] if r["kind"] == "single model"),
                     key=lambda r: r["total_cost"])
    baseline = next((r for r in payload["policies"] if r["name"] == "baseline heuristic"), None)

    fig = plt.figure(figsize=(13.5, 6.6), facecolor=PALETTE["surface"])
    grid = GridSpec(1, 2, width_ratios=[1.62, 1.0], wspace=0.16, figure=fig)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_facecolor(PALETTE["surface"])

    # --- recessive frame -----------------------------------------------------
    ax.grid(True, which="major", color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])

    # --- uncertainty band, then the frontier on top --------------------------
    band = [uncertainty.get(r["name"]) for r in front]
    if all(band):
        ax.fill_between([r["total_cost"] for r in front],
                        [b["quality_lo"] for b in band], [b["quality_hi"] for b in band],
                        color=PALETTE["router"], alpha=0.14, linewidth=0, zorder=2,
                        label="95% interval (estimator refit)")
    ax.plot([r["total_cost"] for r in sweep], [r["mean_quality"] for r in sweep],
            color=PALETTE["router"], alpha=0.30, linewidth=1.0, zorder=3)
    ax.plot([r["total_cost"] for r in front], [r["mean_quality"] for r in front],
            color=PALETTE["router"], linewidth=2.0, marker="o", markersize=4.5,
            markeredgecolor=PALETTE["surface"], markeredgewidth=0.8, zorder=5,
            label="router frontier")

    # --- single-model policies: routable filled, thin-support hollow ---------
    for row in singles:
        routable = row["name"].replace("always ", "") in candidates
        ax.scatter([row["total_cost"]], [row["mean_quality"]], s=54, marker="s",
                   facecolor=PALETTE["single"] if routable else PALETTE["surface"],
                   edgecolor=PALETTE["single"], linewidths=1.4, zorder=6)

    if baseline:
        ax.scatter([baseline["total_cost"]], [baseline["mean_quality"]], s=78, marker="D",
                   facecolor=PALETTE["baseline"], edgecolor=PALETTE["surface"],
                   linewidths=1.2, zorder=7, label="repo baseline heuristic")

    ax.scatter([logged["total_cost"]], [logged["mean_quality"]], s=230, marker="*",
               facecolor=PALETTE["ink"], edgecolor=PALETTE["surface"], linewidths=1.0,
               zorder=8, label="logged policy")

    # --- selective direct labels: only the two points anyone will quote ------
    cheaper = [r for r in sweep if r["mean_quality"] >= logged["mean_quality"]]
    if cheaper:
        point = min(cheaper, key=lambda r: r["total_cost"])
        ax.annotate(f"{point['cost_vs_logged']:+.0%} cost,\nquality held",
                    (point["total_cost"], point["mean_quality"]),
                    textcoords="offset points", xytext=(-14, -34), ha="right", fontsize=8.5,
                    color=PALETTE["ink_soft"], linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color=PALETTE["grid"], linewidth=1.1))
    same_cost = [r for r in sweep if r["total_cost"] <= logged["total_cost"]]
    if same_cost:
        point = max(same_cost, key=lambda r: r["mean_quality"])
        ax.annotate(f"{point['quality_vs_logged']:+.4f} quality,\n{point['cost_vs_logged']:+.0%} cost",
                    (point["total_cost"], point["mean_quality"]),
                    textcoords="offset points", xytext=(2, 34), ha="center", fontsize=8.5,
                    color=PALETTE["ink_soft"], linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color=PALETTE["grid"], linewidth=1.1))
    ax.annotate("logged", (logged["total_cost"], logged["mean_quality"]),
                textcoords="offset points", xytext=(11, -11), fontsize=8.5,
                color=PALETTE["ink"], fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("estimated cost, held-out split (USD, log scale)", fontsize=9,
                  color=PALETTE["ink_soft"], labelpad=8)
    ax.set_ylabel("estimated tool-call success", fontsize=9, color=PALETTE["ink_soft"],
                  labelpad=8)
    ax.tick_params(colors=PALETTE["ink_soft"], labelsize=8.5)
    ax.margins(x=0.14, y=0.20)
    # upper left is the only quadrant with no marks in it
    legend = ax.legend(loc="upper left", fontsize=8.5, frameon=True,
                       facecolor=PALETTE["surface"], edgecolor=PALETTE["grid"])
    legend.set_zorder(10)
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_soft"])

    # --- the table carries every identity ------------------------------------
    table_ax = fig.add_subplot(grid[0, 1])
    table_ax.axis("off")

    lines = [("policy", "cost", "vs log", "quality", "\u0394 qual")]
    ordering = sorted(
        [logged] + ([baseline] if baseline else []) + list(singles),
        key=lambda r: r["total_cost"],
    )
    for row in ordering:
        name = row["name"].replace("always ", "")
        if name.replace(" ", "") and row["kind"] == "single model" and name not in candidates:
            name += " *"
        lines.append((
            name,
            f"${row['total_cost']:,.2f}",
            f"{row['cost_vs_logged']:+.0%}",
            f"{row['mean_quality']:.4f}",
            f"{row['quality_vs_logged']:+.4f}",
        ))

    y = 0.98
    for index, cells in enumerate(lines):
        header = index == 0
        is_logged = cells[0] == "logged policy"
        colour = PALETTE["ink"] if (header or is_logged) else PALETTE["ink_soft"]
        weight = "bold" if (header or is_logged) else "normal"
        for x, cell, align in zip((0.0, 0.52, 0.68, 0.83, 0.99), cells,
                                  ("left", "right", "right", "right", "right")):
            table_ax.text(x, y, cell, fontsize=8.3, color=colour, fontweight=weight,
                          ha=align, va="top", family="DejaVu Sans Mono", transform=table_ax.transAxes)
        y -= 0.062
        if header:
            table_ax.plot([0, 1], [y + 0.030, y + 0.030], color=PALETTE["grid"],
                          linewidth=0.9, transform=table_ax.transAxes, clip_on=False)
    table_ax.text(0.0, y - 0.02,
                  "* excluded from routing: too few logged requests\n"
                  "Prices assumed; input tokens only; quality is a\ntool-call-success proxy, not task success.",
                  fontsize=7.6, color=PALETTE["ink_soft"], va="top",
                  transform=table_ax.transAxes, linespacing=1.5)

    fig.suptitle(f"What routing saves \u2014 {payload['n_requests']} held-out requests",
                 fontsize=13, color=PALETTE["ink"], x=0.5, y=0.98, ha="center")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["surface"])
    return fig


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .config import accurate_config, fast_config
    from .evaluate import format_metrics_table
    from .pipeline import run_pipeline

    parser = argparse.ArgumentParser(description="Cost-quality frontier for the router.")
    parser.add_argument("--encoder", choices=("static", "minilm"), default="static")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT,
                        help="minimum training requests before a model may be routed to")
    args = parser.parse_args(argv)

    config = fast_config() if args.encoder == "static" else accurate_config()
    report = run_pipeline(config, verbose=False, save=False)
    payload = build_frontier(report, config, split=args.split, min_support=args.min_support)

    print()
    print(format_metrics_table(frontier_table(payload)))

    output = config.results_dir / "frontier.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=float))
    try:
        plot_frontier(payload, path=config.results_dir / "frontier.png")
    except ImportError:
        print("matplotlib not installed; skipped the chart")
    print(f"\nwrote {output} and frontier.png")


if __name__ == "__main__":
    main()
