#!/usr/bin/env python3
"""Evaluate a frozen router artifact on a completely external data chunk.

The script does not refit either model, select a new tolerance, or alter the
artifact. Observed workload is calibrated against the artifact's training
distribution, and quality uses only the automatic telemetry proxy available in
the external chunk.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from train_two_stage_router import (
    attach_complexity_signals,
    evaluate,
    observed_complexity,
    request_signals,
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(summary):
    result = summary["result"]
    return f"""# Frozen chunk-01 router on external chunk 02

## Experimental contract

- The router artifact was trained and tuned on chunk 01 only.
- All **{summary['external_rows']}** chunk-02 trajectories were kept outside training and tolerance selection.
- **{summary['usable_rows']}** trajectories had usable automatic telemetry outcomes.
- The workload target uses the frozen chunk-01 percentile calibration.
- No chunk-02 manual labels were introduced.

## External result

- Workload prediction: MAE **{result['complexity_mae']:.2f}**, RMSE **{result['complexity_rmse']:.2f}**, correlation **{result['complexity_correlation']:.3f}**.
- Execution-risk prediction: Brier **{result['execution_risk_brier']:.3f}**, ROC AUC **{result['execution_risk_auc']:.3f}** at prevalence **{result['execution_risk_prevalence']:.1%}**.
- Routes changed: **{result['changed_routes']}/{result['holdout_n']}**; selected actions: **{result['route_counts']}**.
- Estimated opening-input cost: **${result['incumbent_opening_cost_usd_est']:.4f} -> ${result['policy_opening_cost_usd_est']:.4f}**, saving **{result['cost_savings_pct_est']:.1%}**.
- Stabilized doubly robust quality: **{result['observed_quality']:.3f} -> {result['dr_policy_quality']:.3f}**; delta **{result['dr_quality_delta']:+.3f}**, bootstrap 95% CI **[{result['dr_quality_delta_ci_low']:+.3f}, {result['dr_quality_delta_ci_high']:+.3f}]**.
- Policy/behavior matches: **{result['matched_actions']}**; overlap ESS: **{result['overlap_effective_sample_size']:.1f}**.

## Interpretation

This is the cleanest available generalization check because chunk 02 did not
influence training or policy selection. Quality is still an off-policy estimate
from telemetry proxies, not randomized counterfactual evidence, and the missing
final response limits semantic outcome validation.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--artifact", default="models/two_stage_router.json")
    parser.add_argument("--details", default="results/chunk02_external/test_details.csv")
    parser.add_argument("--summary-json", default="results/chunk02_external/summary.json")
    parser.add_argument("--summary-md", default="results/chunk02_external/summary.md")
    parser.add_argument("--propensity-floor", type=float, default=0.03)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    artifact_bytes = artifact_path.read_bytes()
    artifact = json.loads(artifact_bytes)
    if "complexity_calibrator" not in artifact:
        raise SystemExit("artifact lacks its frozen training complexity calibrator")

    with Path(args.evaluation).open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    eligible = [row for row in all_rows if row["eligible_for_modeling"] == "1"]
    usable = []
    for source in eligible:
        if source["y_outcome_label"] == "UNKNOWN":
            continue
        row = dict(source)
        row["evaluation_outcome_label"] = row["y_outcome_label"]
        row["evaluation_outcome_score"] = row["y_outcome_score"]
        row["evaluation_label_confidence"] = row["y_label_confidence"]
        row["evaluation_label_source"] = "telemetry_proxy_external"
        row["evaluation_outcome_usable"] = "1"
        usable.append(row)

    enriched = attach_complexity_signals(usable, request_signals(args.export))
    calibrator = artifact["complexity_calibrator"]
    targets = {
        row["trajectory_id"]: observed_complexity(row, calibrator)
        for row in enriched
    }
    eval_args = SimpleNamespace(
        min_stratum=int(artifact["minimum_exact_stratum"]),
        propensity_floor=args.propensity_floor,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    result, details = evaluate(
        enriched,
        artifact,
        targets,
        eval_args,
        float(artifact["quality_tolerance"]),
    )
    summary = {
        "evaluation_kind": "external_chunk_holdout",
        "training_chunks": ["trajectories_v1_01.jsonl"],
        "external_chunk": "trajectories_v1_02.jsonl",
        "external_rows": len(eligible),
        "usable_rows": len(enriched),
        "outcome_label_source": "automatic telemetry proxy only",
        "artifact_path": str(artifact_path),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "artifact_training_rows": artifact.get("training_rows"),
        "quality_tolerance_frozen": artifact["quality_tolerance"],
        "result": result,
    }

    write_csv(args.details, details)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Path(args.summary_md).write_text(markdown_summary(summary), encoding="utf-8")
    print(
        f"external rows={len(eligible)} usable={len(enriched)} "
        f"saving={result['cost_savings_pct_est']:.1%} "
        f"quality_delta={result['dr_quality_delta']:+.4f} "
        f"CI=[{result['dr_quality_delta_ci_low']:+.4f}, "
        f"{result['dr_quality_delta_ci_high']:+.4f}]"
    )
    print(f"wrote {args.details}, {args.summary_json}, {args.summary_md}")


if __name__ == "__main__":
    main()
