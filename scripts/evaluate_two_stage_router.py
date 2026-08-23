#!/usr/bin/env python3
"""Complete same-test evaluation of the two-stage router.

Produces:
- an incumbent / corrected-heuristic / learned-policy comparison;
- the learned cost-quality frontier with bootstrap intervals;
- sensitivity to alternative benchmark-prior mappings;
- a concise evaluation summary.

The final test split and all nuisance models are reconstructed exactly as in
`train_two_stage_router.py`. The selected epsilon remains validation-selected;
the test frontier is diagnostic and is not used to retune it.
"""

import argparse
import copy
import csv
import html
import json
import statistics
from collections import Counter
from pathlib import Path

from cost_model import load_pricing
from train_two_stage_router import (
    attach_complexity_signals,
    canonical_action,
    choose_action,
    estimated_opening_cost,
    evaluate,
    effective_sample_size,
    fit_artifact,
    fit_complexity_calibrator,
    observed_complexity,
    observed_execution_risk,
    out_of_fold_stage1,
    predict_complexity,
    predict_execution_risk,
    predict_quality,
    propensity,
    request_signals,
    stabilized_dr_value,
    bootstrap_stabilized_delta,
    stratified_three_way,
)


TOLERANCES = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
CLAUDE_ACTIONS = {"fable", "sonnet", "opus"}
PROFILE_DIMENSIONS = (
    "tool_use", "coding", "reasoning", "instruction_following", "long_context", "multimodal"
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_frontier_svg(path, comparison):
    width, height = 960, 560
    left, right, top, bottom = 92, 70, 58, 72
    plot_width, plot_height = width - left - right, height - top - bottom
    visible = comparison
    x_values = [row["policy_cost_usd_est"] for row in visible]
    y_values = []
    for row in visible:
        baseline = row["observed_quality"]
        y_values.extend((
            baseline + row["quality_delta_ci_low"],
            baseline + row["quality_delta_ci_high"],
            row["policy_quality_est"],
        ))
    x_min, x_max = min(x_values), max(x_values)
    x_pad = max(1.0, (x_max - x_min) * 0.08)
    x_min, x_max = max(0.0, x_min - x_pad), x_max + x_pad
    y_min, y_max = min(y_values), max(y_values)
    y_pad = max(0.015, (y_max - y_min) * 0.08)
    y_min, y_max = max(0.0, y_min - y_pad), min(1.02, y_max + y_pad)

    def x(value):
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y(value):
        return top + (y_max - value) / (y_max - y_min) * plot_height

    learned = [row for row in visible if row["policy_type"] == "learned"]
    learned_path = " ".join(
        f"{'M' if index == 0 else 'L'} {x(row['policy_cost_usd_est']):.1f} {y(row['policy_quality_est']):.1f}"
        for index, row in enumerate(learned)
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Cost-quality frontier on the final test set</title>',
        '<desc id="desc">Estimated opening input cost versus doubly robust quality, with bootstrap confidence intervals.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="92" y="30" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#171717">Cost–quality frontier · final test</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fbfbfd" stroke="#d8d8df"/>',
    ]
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e7e7ec"/>')
        parts.append(f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#555">{value:.2f}</text>')
    for index in range(6):
        value = x_min + (x_max - x_min) * index / 5
        xx = x(value)
        anchor = "start" if index == 0 else "end" if index == 5 else "middle"
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+24}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="12" fill="#555">${value:.0f}</text>')
    parts.extend([
        f'<text x="{left + plot_width/2:.1f}" y="{height-18}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#333">Estimated opening-input cost on {visible[0]["test_n"]} template-held-out trajectories (lower is better)</text>',
        f'<text transform="translate(22 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#333">Estimated outcome quality (higher is better)</text>',
        f'<path d="{learned_path}" fill="none" stroke="#6748fd" stroke-width="2.5" opacity="0.75"/>',
    ])
    for row in visible:
        xx = x(row["policy_cost_usd_est"])
        center = row["policy_quality_est"]
        low = row["observed_quality"] + row["quality_delta_ci_low"]
        high = row["observed_quality"] + row["quality_delta_ci_high"]
        yy, y_low, y_high = y(center), y(low), y(high)
        if row["policy_type"] == "incumbent":
            color, radius = "#222222", 6
        elif row["policy_type"] == "heuristic":
            color, radius = "#e06b27", 6
        elif row["validation_selected"]:
            color, radius = "#0b8f6a", 8
        else:
            color, radius = "#6748fd", 5
        if row["policy_type"] != "incumbent":
            parts.extend([
                f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y_high:.1f}" y2="{y_low:.1f}" stroke="{color}" stroke-width="1.5" opacity="0.7"/>',
                f'<line x1="{xx-5:.1f}" x2="{xx+5:.1f}" y1="{y_high:.1f}" y2="{y_high:.1f}" stroke="{color}"/>',
                f'<line x1="{xx-5:.1f}" x2="{xx+5:.1f}" y1="{y_low:.1f}" y2="{y_low:.1f}" stroke="{color}"/>',
            ])
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{radius}" fill="{color}" stroke="#fff" stroke-width="2"/>')
        if row["policy_type"] in {"incumbent", "heuristic"} or row["validation_selected"]:
            label = row["policy"].replace("Learned epsilon=", "Learned ε=")
            label = label.replace("Incumbent logged route", "Incumbent")
            label = label.replace("Public-tier 15k heuristic", "15k heuristic")
            anchor = "end" if row["policy_type"] == "incumbent" else "start"
            dx = -10 if anchor == "end" else 10
            parts.append(
                f'<text x="{xx+dx:.1f}" y="{yy-10:.1f}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="12" font-weight="600" fill="{color}">{html.escape(label)}</text>'
            )
    parts.extend([
        '</svg>',
    ])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def policy_metrics(test, artifact, targets, chooser, label, policy_type, selected, args):
    details = []
    changed = 0
    incumbent_cost = 0.0
    policy_cost = 0.0
    route_counts = Counter()
    chosen_by_id = {}
    for row in test:
        incumbent = canonical_action(row["logged_model"])
        complexity = predict_complexity(row, artifact["complexity_model"])
        chosen = chooser(row, complexity)
        observed = float(row["evaluation_outcome_score"])
        risk = predict_execution_risk(row, artifact["complexity_model"])
        direct_quality = predict_quality(row, complexity, chosen, artifact, risk)
        matched = chosen == incumbent
        inverse_propensity = (
            1.0 / propensity(row, incumbent, artifact, args.propensity_floor)
            if matched else 0.0
        )
        observed_model_quality = predict_quality(
            row, complexity, incumbent, artifact, risk
        )
        details.append({
            "direct_policy_quality": direct_quality,
            "observed_quality": observed,
            "matched_observed_action": int(matched),
            "inverse_propensity": inverse_propensity,
            "outcome_residual": observed - observed_model_quality,
        })
        changed += chosen != incumbent
        incumbent_cost += estimated_opening_cost(row, incumbent, artifact)
        policy_cost += estimated_opening_cost(row, chosen, artifact)
        route_counts[chosen] += 1
        chosen_by_id[row["trajectory_id"]] = chosen
    observed_quality = statistics.fmean(float(row["evaluation_outcome_score"]) for row in test)
    if policy_type == "incumbent":
        policy_quality = observed_quality
        policy_quality_unbounded = observed_quality
        delta = 0.0
        ci_low = ci_high = 0.0
    else:
        policy_quality, policy_quality_unbounded = stabilized_dr_value(details)
        delta = policy_quality - observed_quality
        ci_low, ci_high = bootstrap_stabilized_delta(details, args.bootstrap, args.seed)
    return {
        "policy": label,
        "policy_type": policy_type,
        "validation_selected": int(selected),
        "test_n": len(test),
        "changed_routes": changed,
        "incumbent_cost_usd_est": incumbent_cost,
        "policy_cost_usd_est": policy_cost,
        "cost_savings_pct_est": (incumbent_cost - policy_cost) / incumbent_cost,
        "observed_quality": observed_quality,
        "direct_quality_est": statistics.fmean(
            row["direct_policy_quality"] for row in details
        ),
        "policy_quality_est": policy_quality,
        "policy_quality_unbounded": policy_quality_unbounded,
        "quality_delta_est": delta,
        "quality_delta_ci_low": ci_low,
        "quality_delta_ci_high": ci_high,
        "matched_actions": sum(row["matched_observed_action"] for row in details),
        "overlap_effective_sample_size": effective_sample_size(details),
        "route_counts": json.dumps(dict(route_counts), sort_keys=True),
        "_chosen": chosen_by_id,
        "_details": details,
    }


def clipped_stabilized_value(details, cap):
    """Stabilized DR diagnostic with optional importance-weight clipping."""
    direct = statistics.fmean(row["direct_policy_quality"] for row in details)
    matched = [row for row in details if row["matched_observed_action"]]
    weights = [
        min(row["inverse_propensity"], cap) if cap is not None
        else row["inverse_propensity"]
        for row in matched
    ]
    correction = (
        sum(weight * row["outcome_residual"] for weight, row in zip(weights, matched))
        / sum(weights)
        if weights else 0.0
    )
    raw = direct + correction
    ess = (sum(weights) ** 2 / sum(weight * weight for weight in weights)) if weights else 0.0
    return min(1.0, max(0.0, raw)), ess, max(weights, default=0.0)


def learned_chooser(artifact, tolerance, min_stratum):
    def choose(row, _complexity):
        return choose_action(row, artifact, tolerance, min_stratum)[0]
    return choose


def heuristic_chooser(artifact, min_stratum, threshold=15_000):
    def choose(row, _complexity):
        incumbent = canonical_action(row["logged_model"])
        if float(row["x_total_context_tokens_est"]) >= threshold:
            return incumbent
        target = "sonnet" if incumbent in CLAUDE_ACTIONS else "luna"
        from train_two_stage_router import action_supported
        return target if action_supported(row, target, artifact, min_stratum)[0] else incumbent
    return choose


def transform_profiles(config, scenario):
    transformed = copy.deepcopy(config)
    profiles = transformed["profiles"]
    if scenario == "base":
        return transformed
    if scenario in {"compressed", "expanded"}:
        factor = 0.5 if scenario == "compressed" else 1.25
        center = 0.80
        for profile in profiles.values():
            for dimension in PROFILE_DIMENSIONS:
                profile[dimension] = min(1.0, max(0.0, center + factor * (profile[dimension] - center)))
        transformed["description"] += f" Scenario: {scenario} capability separation."
        return transformed
    if scenario == "gpt_sol_luna_swapped":
        sol = {dimension: profiles["sol"][dimension] for dimension in PROFILE_DIMENSIONS}
        luna = {dimension: profiles["luna"][dimension] for dimension in PROFILE_DIMENSIONS}
        for dimension in PROFILE_DIMENSIONS:
            profiles["sol"][dimension] = luna[dimension]
            profiles["luna"][dimension] = sol[dimension]
        transformed["description"] += " Scenario: reverse the most uncertain GPT Sol/Luna capability mapping."
        return transformed
    raise ValueError(f"unknown sensitivity scenario: {scenario}")


def strip_internal(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def markdown_summary(comparison, sensitivity, ope_sensitivity, selected_tolerance):
    selected = next(row for row in comparison if row["validation_selected"])
    heuristic = next(row for row in comparison if row["policy"] == "Public-tier 15k heuristic")
    return f"""# Completed two-stage router evaluation

## Same-test comparison

The validation-selected learned policy uses epsilon **{selected_tolerance:.3f}**. On the **{selected['test_n']}** template-held-out test trajectories it saves **{selected['cost_savings_pct_est']:.1%}** of estimated opening-input cost with a stabilized quality delta of **{selected['quality_delta_est']:+.3f}** (95% CI **[{selected['quality_delta_ci_low']:+.3f}, {selected['quality_delta_ci_high']:+.3f}]**). Its direct-model estimate is **{selected['direct_quality_est']:.3f}** and overlap ESS is **{selected['overlap_effective_sample_size']:.1f}**.

The corrected 15k heuristic saves **{heuristic['cost_savings_pct_est']:.1%}** with a stabilized quality delta of **{heuristic['quality_delta_est']:+.3f}** (95% CI **[{heuristic['quality_delta_ci_low']:+.3f}, {heuristic['quality_delta_ci_high']:+.3f}]**) on the same rows and with the same nuisance models; overlap ESS is **{heuristic['overlap_effective_sample_size']:.1f}**.

## Benchmark-prior sensitivity

{chr(10).join(f"- `{row['scenario']}`: saving {row['cost_savings_pct_est']:.1%}, quality delta {row['quality_delta_est']:+.3f} [{row['quality_delta_ci_low']:+.3f}, {row['quality_delta_ci_high']:+.3f}], route disagreement vs base {row['route_disagreement_vs_base']:.1%}." for row in sensitivity)}

## Importance-weight sensitivity

{chr(10).join(f"- cap `{row['weight_cap']}`: quality delta {row['quality_delta_est']:+.3f}, overlap ESS {row['overlap_effective_sample_size']:.1f}, maximum effective weight {row['maximum_effective_weight']:.1f}." for row in ope_sensitivity)}

## Interpretation

The learned router is evaluated with stabilized rather than raw inverse-propensity corrections, and price is excluded from its quality model. The quality interval still crosses zero, so the result supports a cost-saving claim with no detected quality change—not a proven quality improvement. The benchmark sensitivity table tests whether the result is driven by capability-prior spacing or the uncertain GPT Sol/Luna ordering. All quality comparisons remain off-policy estimates based largely on telemetry proxy labels.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="results/calibrated_evaluation_table.csv")
    parser.add_argument("--export", default="export")
    parser.add_argument("--profiles", default="models/benchmark_priors.json")
    parser.add_argument("--comparison", default="results/evaluator_comparison.csv")
    parser.add_argument("--sensitivity", default="results/benchmark_sensitivity.csv")
    parser.add_argument("--ope-sensitivity", default="results/ope_weight_sensitivity.csv")
    parser.add_argument("--summary", default="results/evaluator_summary.md")
    parser.add_argument("--chart", default="results/cost_quality_frontier.svg")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--complexity-penalty", type=float, default=150.0)
    parser.add_argument("--text-features", type=int, default=96)
    parser.add_argument("--quality-penalty", type=float, default=15.0)
    parser.add_argument("--quality-tolerance", type=float)
    parser.add_argument(
        "--heuristic-thresholds",
        default="5000,10000,15000,20000,30000,50000",
        help="Comma-separated opening-token thresholds for the diagnostic heuristic sweep.",
    )
    parser.add_argument("--caliper-quantile", type=float, default=0.95)
    parser.add_argument("--min-stratum", type=int, default=3)
    parser.add_argument("--fallback-action", default="sonnet")
    parser.add_argument("--propensity-floor", type=float, default=0.03)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.quality_tolerance is None:
        summary_path = Path("results/two_stage_summary.json")
        args.quality_tolerance = (
            json.loads(summary_path.read_text(encoding="utf-8"))["selected_quality_tolerance"]
            if summary_path.exists() else 0.025
        )

    with Path(args.evaluation).open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["evaluation_outcome_usable"] == "1"]
    enriched = attach_complexity_signals(rows, request_signals(args.export))
    train, validation, test = stratified_three_way(
        enriched, args.validation_fraction, args.test_fraction
    )
    development = train + validation
    calibrator = fit_complexity_calibrator(development)
    targets = {
        row["trajectory_id"]: observed_complexity(row, calibrator)
        for row in development + test
    }
    risk_targets = {
        row["trajectory_id"]: observed_execution_risk(row)
        for row in development + test
    }
    oof_stage1 = out_of_fold_stage1(
        development, targets, risk_targets,
        args.complexity_penalty, args.oof_folds, args.text_features,
    )
    profiles_config = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    pricing = load_pricing()
    base_artifact = fit_artifact(
        development, targets, risk_targets, oof_stage1,
        profiles_config, pricing, args,
    )

    comparison = []
    comparison.append(policy_metrics(
        test, base_artifact, targets,
        lambda row, complexity: canonical_action(row["logged_model"]),
        "Incumbent logged route", "incumbent", False, args,
    ))
    heuristic_thresholds = sorted({
        int(value) for value in args.heuristic_thresholds.split(",") if value.strip()
    })
    for threshold in heuristic_thresholds:
        comparison.append(policy_metrics(
            test, base_artifact, targets,
            heuristic_chooser(base_artifact, args.min_stratum, threshold),
            f"Public-tier {threshold / 1000:g}k heuristic",
            "heuristic", False, args,
        ))
    for tolerance in TOLERANCES:
        comparison.append(policy_metrics(
            test, base_artifact, targets,
            learned_chooser(base_artifact, tolerance, args.min_stratum),
            f"Learned epsilon={tolerance:.3f}", "learned",
            tolerance == args.quality_tolerance, args,
        ))

    base_selected = next(row for row in comparison if row["validation_selected"])
    base_choices = base_selected["_chosen"]
    ope_sensitivity = []
    for cap in (2.0, 5.0, 10.0, 20.0, None):
        quality, ess, maximum_weight = clipped_stabilized_value(
            base_selected["_details"], cap
        )
        ope_sensitivity.append({
            "weight_cap": "unclipped" if cap is None else cap,
            "policy_quality_est": quality,
            "quality_delta_est": quality - base_selected["observed_quality"],
            "overlap_effective_sample_size": ess,
            "maximum_effective_weight": maximum_weight,
        })
    sensitivity = []
    scenarios = (
        ("base", "Configured benchmark priors"),
        ("compressed", "Capability gaps compressed 50% toward 0.80"),
        ("expanded", "Capability gaps expanded 25% around 0.80"),
        ("gpt_sol_luna_swapped", "Sol and Luna capability profiles swapped"),
    )
    for scenario, description in scenarios:
        config = transform_profiles(profiles_config, scenario)
        artifact = fit_artifact(
            development, targets, risk_targets, oof_stage1, config, pricing, args
        )
        result = policy_metrics(
            test, artifact, targets,
            learned_chooser(artifact, args.quality_tolerance, args.min_stratum),
            f"Sensitivity: {scenario}", "sensitivity", scenario == "base", args,
        )
        disagreement = statistics.fmean(
            result["_chosen"][trajectory_id] != action
            for trajectory_id, action in base_choices.items()
        )
        sensitivity.append({
            "scenario": scenario,
            "description": description,
            "test_n": len(test),
            "changed_routes": result["changed_routes"],
            "cost_savings_pct_est": result["cost_savings_pct_est"],
            "quality_delta_est": result["quality_delta_est"],
            "quality_delta_ci_low": result["quality_delta_ci_low"],
            "quality_delta_ci_high": result["quality_delta_ci_high"],
            "direct_quality_est": result["direct_quality_est"],
            "matched_actions": result["matched_actions"],
            "overlap_effective_sample_size": result["overlap_effective_sample_size"],
            "route_disagreement_vs_base": disagreement,
            "route_counts": result["route_counts"],
        })

    write_csv(args.comparison, [strip_internal(row) for row in comparison])
    write_csv(args.sensitivity, sensitivity)
    write_csv(args.ope_sensitivity, ope_sensitivity)
    write_frontier_svg(args.chart, comparison)
    Path(args.summary).write_text(
        markdown_summary(
            comparison, sensitivity, ope_sensitivity, args.quality_tolerance
        ),
        encoding="utf-8",
    )

    selected = base_selected
    heuristic = next(row for row in comparison if row["policy"] == "Public-tier 15k heuristic")
    print(
        f"selected learned: saving={selected['cost_savings_pct_est']:.1%} "
        f"quality_delta={selected['quality_delta_est']:+.4f} "
        f"CI=[{selected['quality_delta_ci_low']:+.4f}, {selected['quality_delta_ci_high']:+.4f}]"
    )
    print(
        f"heuristic: saving={heuristic['cost_savings_pct_est']:.1%} "
        f"quality_delta={heuristic['quality_delta_est']:+.4f}"
    )
    print(f"wrote {args.comparison}, {args.sensitivity}, {args.summary}, {args.chart}")


if __name__ == "__main__":
    main()
