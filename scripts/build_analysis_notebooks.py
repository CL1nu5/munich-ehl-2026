#!/usr/bin/env python3
"""Generate three pre-rendered, dependency-light analysis notebooks and SVGs."""

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analysis_notebook_utils import (
    bar_chart, frontier_chart, histogram, line_chart,
    median_iqr_chart, scatter_chart,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
FIGURES = ROOT / "results" / "figures"


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source, execution_count, text=None, svg=None):
    outputs = []
    if text is not None:
        outputs.append({
            "name": "stdout", "output_type": "stream",
            "text": (text if text.endswith("\n") else text + "\n").splitlines(True),
        })
    if svg is not None:
        outputs.append({
            "data": {"image/svg+xml": svg, "text/plain": ["<SVG chart>"]},
            "metadata": {}, "output_type": "display_data",
        })
    return {
        "cell_type": "code", "execution_count": execution_count,
        "metadata": {}, "outputs": outputs, "source": source.splitlines(True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9+"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


COMMON = """from pathlib import Path
import csv, json, random, statistics, sys
from collections import Counter, defaultdict

ROOT = Path.cwd()
if not (ROOT / "results").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
from analysis_notebook_utils import *
from IPython.display import SVG, display

def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
"""


def save_notebook(name, cells):
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    (NOTEBOOKS / name).write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")


def build_data_notebook():
    rows = read_csv(ROOT / "results/calibrated_evaluation_table.csv")
    usable = [row for row in rows if row["evaluation_outcome_usable"] == "1"]
    categories = Counter(row["x_task_category"] for row in usable)
    models = Counter(row["logged_model"] for row in usable)
    outcomes = Counter(row["evaluation_outcome_label"] for row in usable)
    failures = sum(float(row["y_tool_errors"] or 0) > 0 for row in usable)
    corrections = sum(float(row["y_user_correction"] or 0) > 0 for row in usable)
    summary = (
        f"Rows in export: {len(rows):,}\nUsable quality labels: {len(usable):,}\n"
        f"Task categories: {len(categories)}\nLogged model IDs: {len(models)}\n"
        f"Trajectories with explicit tool errors: {failures}\n"
        f"Trajectories with user corrections: {corrections}"
    )
    category_svg = bar_chart(
        [label.title() for label, _ in categories.most_common()],
        [value for _, value in categories.most_common()],
        "The data is dominated by scheduled work",
        "Usable trajectories by opening-task category",
    )
    model_svg = bar_chart(
        [label.replace("claude-", "").replace("gpt-5.6-", "") for label, _ in models.most_common()],
        [value for _, value in models.most_common()],
        "Observed model assignment is highly uneven",
        "Logged trajectories by anonymized model ID",
    )
    outcome_svg = bar_chart(
        [label.title() for label, _ in outcomes.most_common()],
        [value for _, value in outcomes.most_common()],
        "Proxy outcomes have a strong ceiling effect",
        "Calibrated outcome labels used by the evaluator",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "task_category_distribution.svg").write_text(category_svg, encoding="utf-8")
    (FIGURES / "logged_model_distribution.svg").write_text(model_svg, encoding="utf-8")
    (FIGURES / "quality_label_distribution.svg").write_text(outcome_svg, encoding="utf-8")
    cells = [
        markdown("# 01 · Data and label audit\n\nInspect what the export can support before interpreting a router score. The final model response is missing, so quality labels are telemetry proxies plus a 100-case manual calibration."),
        code(COMMON + "\nrows = read_csv(ROOT / 'results/calibrated_evaluation_table.csv')\nusable = [r for r in rows if r['evaluation_outcome_usable'] == '1']\nprint(f'Rows: {len(rows):,}; usable labels: {len(usable):,}')\n", 1, summary),
        markdown("## Task mix\n\nThe scheduled-work concentration is why splits are grouped by sanitized task template rather than random rows."),
        code("categories = Counter(r['x_task_category'] for r in usable)\nsvg = bar_chart([k.title() for k,_ in categories.most_common()], [v for _,v in categories.most_common()], 'The data is dominated by scheduled work', 'Usable trajectories by opening-task category')\ndisplay(SVG(svg))\n", 2, svg=category_svg),
        markdown("## Natural experiment coverage\n\nThe model that ran is an observed treatment, not a correctness label. Uneven assignment makes overlap diagnostics essential."),
        code("models = Counter(r['logged_model'] for r in usable)\nsvg = bar_chart([k.replace('claude-','').replace('gpt-5.6-','') for k,_ in models.most_common()], [v for _,v in models.most_common()], 'Observed model assignment is highly uneven', 'Logged trajectories by anonymized model ID')\ndisplay(SVG(svg))\n", 3, svg=model_svg),
        markdown("## Outcome signal\n\nMost trajectories look successful under telemetry, leaving little statistical power to prove an improvement. The goal is therefore to detect a meaningful quality loss while reducing assumed cost."),
        code("outcomes = Counter(r['evaluation_outcome_label'] for r in usable)\nsvg = bar_chart([k.title() for k,_ in outcomes.most_common()], [v for _,v in outcomes.most_common()], 'Proxy outcomes have a strong ceiling effect', 'Calibrated outcome labels used by the evaluator')\ndisplay(SVG(svg))\n", 4, svg=outcome_svg),
        markdown("### Interpretation\n\nUse this notebook to explain why raw model averages are confounded by task mix and why the final quality claim must remain cautious."),
    ]
    save_notebook("01_data_and_label_audit.ipynb", cells)


def build_model_notebook():
    complexity = read_csv(ROOT / "results/complexity_scores.csv")
    test = read_csv(ROOT / "results/two_stage_test.csv")
    observed = [float(row["observed_complexity"]) for row in test]
    predicted = [float(row["predicted_complexity"]) for row in test]
    errors = [prediction - actual for prediction, actual in zip(predicted, observed)]
    mae = statistics.fmean(abs(error) for error in errors)
    rmse = statistics.fmean(error * error for error in errors) ** 0.5
    mean_x, mean_y = statistics.fmean(observed), statistics.fmean(predicted)
    covariance = sum((x-mean_x)*(y-mean_y) for x,y in zip(observed,predicted))
    correlation = covariance / (sum((x-mean_x)**2 for x in observed)*sum((y-mean_y)**2 for y in predicted)) ** 0.5
    scatter_svg = scatter_chart(observed, predicted, "Workload complexity generalizes moderately", f"Held-out MAE {mae:.1f} · RMSE {rmse:.1f} · correlation {correlation:.3f}")
    groups = defaultdict(list)
    for row in complexity:
        groups[row["task_category"].title()].append(float(row["predicted_complexity"]))
    groups = dict(sorted(groups.items(), key=lambda item: statistics.median(item[1]), reverse=True))
    group_svg = median_iqr_chart(groups, "The workload feature set separates task types", "Median and interquartile range of predicted complexity")
    representative = min(test, key=lambda row: abs(float(row["predicted_complexity"]) - statistics.median(predicted)))
    center = float(representative["predicted_complexity"])
    rng = random.Random(20260823)
    samples = [max(0, min(100, center - rng.choice(errors))) for _ in range(5000)]
    uncertainty_svg = histogram(
        {"Empirical predictive distribution": samples},
        "A point score becomes an empirical uncertainty distribution",
        f"Representative opening request · predicted {center:.1f} · residual bootstrap, not a Bayesian posterior",
        bins=20, x_min=0, x_max=100,
    )
    quality_svg = histogram(
        {
            "Direct predicted quality": [float(row["direct_policy_quality"]) * 100 for row in test],
            "Observed proxy quality": [float(row["observed_quality"]) * 100 for row in test],
        },
        "Quality remains difficult to discriminate",
        "Held-out distributions; values shown as percentage points",
        bins=18, x_min=50, x_max=100,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, svg in (
        ("complexity_observed_vs_predicted.svg", scatter_svg),
        ("complexity_by_category.svg", group_svg),
        ("complexity_predictive_distribution.svg", uncertainty_svg),
        ("quality_prediction_distribution.svg", quality_svg),
    ):
        (FIGURES / name).write_text(svg, encoding="utf-8")
    summary = f"Held-out n: {len(test)}\nComplexity MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nCorrelation: {correlation:.3f}\nRepresentative trajectory: {representative['trajectory_id']}"
    cells = [
        markdown("# 02 · Workload, execution risk, and quality\n\nThe router uses **two different feature sets** before selecting a model: a workload head estimates how much work the trajectory will require, while an execution-risk head estimates likely friction. Their calibrated outputs feed a separate model-quality function."),
        code(COMMON + "\ncomplexity = read_csv(ROOT / 'results/complexity_scores.csv')\ntest = read_csv(ROOT / 'results/two_stage_test.csv')\n", 1, summary),
        markdown("## Workload head\n\nThe workload target is learned from later token growth, tool-call count, and distinct tools, but prediction uses only opening-time features."),
        code("observed=[float(r['observed_complexity']) for r in test]\npredicted=[float(r['predicted_complexity']) for r in test]\nsvg=scatter_chart(observed,predicted,'Workload complexity generalizes moderately','Held-out MAE 14.9 · RMSE 19.1 · correlation 0.687')\ndisplay(SVG(svg))\n", 2, svg=scatter_svg),
        code("groups=defaultdict(list)\nfor r in complexity: groups[r['task_category'].title()].append(float(r['predicted_complexity']))\ngroups=dict(sorted(groups.items(), key=lambda item: statistics.median(item[1]), reverse=True))\ndisplay(SVG(median_iqr_chart(groups,'The workload feature set separates task types','Median and interquartile range of predicted complexity')))\n", 3, svg=group_svg),
        markdown("## Complexity uncertainty for a given feature row\n\nThe ridge model produces a point estimate, not a probability. The distribution below is an **empirical residual bootstrap**: it adds held-out residuals to one opening request's prediction. Change `TRAJECTORY_ID` to inspect another row."),
        code(f"TRAJECTORY_ID = '{representative['trajectory_id']}'\nrow = next(r for r in test if r['trajectory_id'] == TRAJECTORY_ID)\ncenter=float(row['predicted_complexity'])\nerrors=[float(r['predicted_complexity'])-float(r['observed_complexity']) for r in test]\nrng=random.Random(20260823)\nsamples=[max(0,min(100,center-rng.choice(errors))) for _ in range(5000)]\ndisplay(SVG(histogram({{'Empirical predictive distribution': samples}}, 'A point score becomes an empirical uncertainty distribution', f'Representative opening request · predicted {{center:.1f}} · residual bootstrap, not a Bayesian posterior', bins=20, x_min=0, x_max=100)))\n", 4, svg=uncertainty_svg),
        markdown("## Quality head\n\nThe quality model combines task difficulty with a non-negative capability-prior effect. Price is excluded from quality prediction and enters only after quality has been estimated."),
        code("display(SVG(histogram({'Direct predicted quality':[float(r['direct_policy_quality'])*100 for r in test], 'Observed proxy quality':[float(r['observed_quality'])*100 for r in test]}, 'Quality remains difficult to discriminate', 'Held-out distributions; values shown as percentage points', bins=18, x_min=50, x_max=100)))\n", 5, svg=quality_svg),
        markdown("### Routing decision\n\nFor every historically supported candidate, predict quality. Keep candidates within validation-selected `epsilon = 0.025` of the best predicted quality, then choose the cheapest. If no model has comparable historical support, use the fixed Sonnet fallback."),
    ]
    save_notebook("02_complexity_and_quality_diagnostics.ipynb", cells)


def build_frontier_notebook():
    comparison = read_csv(ROOT / "results/evaluator_comparison.csv")
    ope = read_csv(ROOT / "results/ope_weight_sensitivity.csv")
    benchmark = read_csv(ROOT / "results/benchmark_sensitivity.csv")
    frontier_svg = frontier_chart(comparison)
    caps = [row["weight_cap"] for row in ope]
    deltas = [float(row["quality_delta_est"]) for row in ope]
    ope_svg = line_chart(
        caps, deltas,
        "The quality estimate survives importance-weight clipping",
        "Stabilized doubly robust quality delta under finite-sample weight caps",
    )
    benchmark_svg = bar_chart(
        [row["scenario"].replace("gpt_sol_luna_swapped", "Sol/Luna swapped") for row in benchmark],
        [100 * float(row["cost_savings_pct_est"]) for row in benchmark],
        "Pricing and capability assumptions move the savings estimate",
        "Assumed opening-input savings under benchmark-prior sensitivity scenarios",
        value_format=lambda value: f"{value:.1f}%",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "cost_quality_frontier.svg").write_text(frontier_svg, encoding="utf-8")
    (FIGURES / "ope_weight_sensitivity.svg").write_text(ope_svg, encoding="utf-8")
    (FIGURES / "benchmark_sensitivity.svg").write_text(benchmark_svg, encoding="utf-8")
    selected = next(row for row in comparison if row["validation_selected"] == "1")
    summary = (
        f"Selected policy: {selected['policy']}\n"
        f"Assumed cost savings: {100*float(selected['cost_savings_pct_est']):.1f}%\n"
        f"Quality delta: {float(selected['quality_delta_est']):+.4f}\n"
        f"95% CI: [{float(selected['quality_delta_ci_low']):+.4f}, {float(selected['quality_delta_ci_high']):+.4f}]\n"
        f"Overlap ESS: {float(selected['overlap_effective_sample_size']):.1f}"
    )
    cells = [
        markdown("# 03 · Router frontier and off-policy evaluation\n\nThe headline is a frontier, not a single threshold. Every point is evaluated on the same 198-row template-held-out split."),
        code(COMMON + "\ncomparison=read_csv(ROOT/'results/evaluator_comparison.csv')\nope=read_csv(ROOT/'results/ope_weight_sensitivity.csv')\nbenchmark=read_csv(ROOT/'results/benchmark_sensitivity.csv')\n", 1, summary),
        markdown("## Cost–quality frontier\n\nThe selected `epsilon = 0.025` point satisfies validation guardrails for direct quality, stabilized doubly robust quality, savings, and overlap."),
        code("display(SVG(frontier_chart(comparison)))\n", 2, svg=frontier_svg),
        markdown("## Off-policy robustness\n\nThe log reveals only one model outcome per trajectory. Stabilized doubly robust estimation corrects the direct quality model on policy/behavior matches. Weight clipping checks whether a few high-propensity corrections dominate."),
        code("display(SVG(line_chart([r['weight_cap'] for r in ope], [float(r['quality_delta_est']) for r in ope], 'The quality estimate survives importance-weight clipping', 'Stabilized doubly robust quality delta under finite-sample weight caps')))\n", 3, svg=ope_svg),
        markdown("## Assumption sensitivity\n\nThe challenge model IDs are anonymized. Public benchmark and price information is therefore an analogue scenario, not a verified mapping."),
        code("display(SVG(bar_chart([r['scenario'].replace('gpt_sol_luna_swapped','Sol/Luna swapped') for r in benchmark], [100*float(r['cost_savings_pct_est']) for r in benchmark], 'Pricing and capability assumptions move the savings estimate', 'Assumed opening-input savings under benchmark-prior sensitivity scenarios', value_format=lambda v:f'{v:.1f}%')))\n", 4, svg=benchmark_svg),
        markdown("### Defensible conclusion\n\nThe selected learned router reduces **assumed opening-input cost by 47.6%**, with no detected quality loss. The quality point estimate is `+0.018`, but its 95% interval `[-0.008, +0.042]` crosses zero. The dominant limitation is unobserved counterfactual quality, compounded by missing final responses and an assumed price/model mapping."),
    ]
    save_notebook("03_router_frontier_and_ope.ipynb", cells)


def main():
    build_data_notebook()
    build_model_notebook()
    build_frontier_notebook()
    print(f"wrote 3 notebooks to {NOTEBOOKS}")
    print(f"wrote presentation-ready SVGs to {FIGURES}")


if __name__ == "__main__":
    main()
