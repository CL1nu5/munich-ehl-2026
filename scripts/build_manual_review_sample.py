#!/usr/bin/env python3
"""Build the manually adjudicated 100-case proxy-calibration sample.

The sample contains all 84 automatic UNKNOWN rows plus 16 model/proxy checks.
Judgments use only observable task/history evidence; missing final outputs remain
UNJUDGEABLE when the history is insufficient.

Usage: python scripts/build_manual_review_sample.py export/
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from build_feature_table import content_text, opening_snapshot, routing_text
from load_trajectories import iter_requests


REVIEW_CHUNK = "trajectories_v1_01.jsonl"
UNKNOWN_PARTIAL = {25, 385, 390}
UNKNOWN_UNJUDGEABLE = {564, 984}

# One SUCCESS and one RECOVERED proxy per sufficiently represented model, with
# manual outcome judgments. Tiny legacy models contribute only their available
# class. The source-line choices are fixed so the sample is reproducible.
PROXY_CHECKS = {
    472: ("SUCCESS", "high", "valid empty-queue no-op completed as instructed"),
    503: ("UNJUDGEABLE", "low", "observable summary refers to prior work and does not resolve the current advice request"),
    956: ("SUCCESS", "high", "digest was sent with the requested sections and state cleanup"),
    506: ("SUCCESS", "high", "requested draft was delivered and ambiguity was surfaced"),
    773: ("PARTIAL", "high", "candidate upload succeeded but suppression checks and outreach were blocked by a disconnected integration"),
    515: ("SUCCESS", "medium", "valid silent heartbeat with no available conversation target"),
    550: ("SUCCESS", "high", "acknowledgment was correctly skipped after live-history verification"),
    275: ("UNJUDGEABLE", "low", "missing surrounding conversation makes the decision to stay silent impossible to verify"),
    445: ("SUCCESS", "medium", "valid no-op after checking email and reminder state"),
    638: ("FAILURE", "high", "a direct addressed question was incorrectly classified as not addressed and left unanswered"),
    535: ("UNJUDGEABLE", "low", "final response is missing and the side-effect evidence alone cannot establish task correctness"),
    416: ("UNJUDGEABLE", "low", "final response is missing and recovered tool execution does not verify semantic correctness"),
    538: ("UNJUDGEABLE", "low", "final response is missing and side-effect telemetry is insufficient for a semantic judgment"),
    467: ("UNJUDGEABLE", "low", "final response is missing for a payroll-file task requiring content validation"),
    392: ("UNJUDGEABLE", "low", "final response is missing and calendar-watch side effects cannot be verified from the excerpt"),
    529: ("SUCCESS", "medium", "valid silent heartbeat after verifying no accessible activity"),
}


def unknown_judgment(line):
    if line in UNKNOWN_PARTIAL:
        reasons = {
            25: "requested deliverable was sent, but an additional canvas update remained blocked by missing Slack scope",
            385: "substantive processing occurred, but a tool error and missing final response prevent complete verification",
            390: "core event work appears complete, but unresolved cleanup/state permission errors and no final response leave partial evidence",
        }
        return "PARTIAL", "medium", reasons[line]
    if line in UNKNOWN_UNJUDGEABLE:
        reasons = {
            564: "no final assistant response and the last observable outputs only show timestamps/logging",
            984: "no final assistant response and the last observable outputs only show checklist cleanup",
        }
        return "UNJUDGEABLE", "low", reasons[line]
    return (
        "SUCCESS",
        "medium",
        "manual review found a completed action or task-valid silent/no-op outcome in observable history",
    )


def excerpts(request):
    opening = opening_snapshot(request)
    opening_len = len(opening.get("input", []))
    history = request.get("input", [])[opening_len:]
    assistant_texts = [
        content_text(item.get("content"))
        for item in history
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    final_assistant = next((text for text in reversed(assistant_texts) if text.strip()), "")
    tool_outputs = [
        str(item.get("output", ""))
        for item in history
        if isinstance(item, dict)
        and item.get("type") in {"function_call_output", "custom_tool_call_output"}
    ]
    return {
        "review_task_excerpt": routing_text(opening.get("input", []))[-700:],
        "review_final_assistant_excerpt": final_assistant[:1200],
        "review_last_tool_output_excerpt": tool_outputs[-1][-1200:] if tool_outputs else "",
    }


def write_summary(rows, output):
    manual = Counter(row["manual_outcome_label"] for row in rows)
    by_source = Counter(row["review_sample_reason"] for row in rows)
    judgeable = [row for row in rows if row["manual_outcome_label"] != "UNJUDGEABLE"]
    proxy_checks = [row for row in rows if row["review_sample_reason"] == "model_proxy_check"]
    proxy_judgeable = [row for row in proxy_checks if row["manual_outcome_label"] != "UNJUDGEABLE"]
    false_positive = sum(row["manual_outcome_label"] == "FAILURE" for row in proxy_judgeable)
    partial = sum(row["manual_outcome_label"] == "PARTIAL" for row in proxy_judgeable)

    text = f"""# Manual proxy-label calibration

## Sample

- Total reviewed: {len(rows)}
- All automatic UNKNOWN cases: {by_source['all_proxy_unknown']}
- Cross-model proxy checks: {by_source['model_proxy_check']}
- Judgeable from observable history: {len(judgeable)}
- Still unjudgeable because the final output/context is missing: {manual['UNJUDGEABLE']}

## Manual outcomes

| Outcome | Count |
|---|---:|
| SUCCESS | {manual['SUCCESS']} |
| PARTIAL | {manual['PARTIAL']} |
| FAILURE | {manual['FAILURE']} |
| UNJUDGEABLE | {manual['UNJUDGEABLE']} |

## Proxy check

Among the {len(proxy_checks)} cross-model automatic SUCCESS/RECOVERED checks, {len(proxy_judgeable)} were manually judgeable. The review found {false_positive} clear failure and {partial} partial completion. The clear failure was a direct user question incorrectly dismissed as “not addressed to me,” demonstrating that successful tool telemetry can coexist with semantic task failure.

This is a calibration sample, not a population estimate: all 84 UNKNOWN cases were intentionally oversampled. The final model output is absent throughout the export, so UNJUDGEABLE must remain a legitimate result.
"""
    output.write_text(text, encoding="utf-8")


def write_calibrated_table(evaluation, sample, output):
    manual_by_source = {
        (row["source_chunk"], row["source_line"]): row for row in sample
    }
    calibrated = []
    for row in evaluation:
        merged = dict(row)
        manual = manual_by_source.get((row["source_chunk"], row["source_line"]))
        if manual:
            label = manual["manual_outcome_label"]
            score = manual["manual_outcome_score"]
            confidence = manual["manual_label_confidence"]
            source = "manual_review"
        else:
            label = row["y_outcome_label"]
            score = row["y_outcome_score"]
            confidence = row["y_label_confidence"]
            source = "telemetry_proxy"
        merged.update({
            "evaluation_outcome_label": label,
            "evaluation_outcome_score": score,
            "evaluation_label_confidence": confidence,
            "evaluation_label_source": source,
            "evaluation_outcome_usable": int(label not in {"UNKNOWN", "UNJUDGEABLE"}),
        })
        calibrated.append(merged)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibrated[0]))
        writer.writeheader()
        writer.writerows(calibrated)
    return calibrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="export")
    parser.add_argument("--evaluation", default="results/evaluation_table.csv")
    parser.add_argument("--output", default="results/manual_review_sample.csv")
    parser.add_argument("--summary", default="results/manual_review_summary.md")
    parser.add_argument("--calibrated", default="results/calibrated_evaluation_table.csv")
    args = parser.parse_args()

    requests = {
        (chunk, line_no + 1): request
        for chunk, line_no, request in iter_requests(args.export)
    }
    with open(args.evaluation, encoding="utf-8", newline="") as handle:
        evaluation = list(csv.DictReader(handle))

    sample = []
    for row in evaluation:
        # The fixed line-number judgments below were made against chunk 01.
        # Never apply them to another chunk that happens to have the same line.
        if row["source_chunk"] != REVIEW_CHUNK:
            continue
        line = int(row["source_line"])
        if row["y_outcome_label"] == "UNKNOWN":
            manual_label, confidence, reason = unknown_judgment(line)
            sample_reason = "all_proxy_unknown"
        elif line in PROXY_CHECKS:
            manual_label, confidence, reason = PROXY_CHECKS[line]
            sample_reason = "model_proxy_check"
        else:
            continue
        reviewed = dict(row)
        reviewed.update(excerpts(requests[(row["source_chunk"], line)]))
        reviewed.update({
            "review_sample_reason": sample_reason,
            "manual_outcome_label": manual_label,
            "manual_outcome_score": {
                "SUCCESS": 1.0, "PARTIAL": 0.5, "FAILURE": 0.0, "UNJUDGEABLE": "",
            }[manual_label],
            "manual_label_confidence": confidence,
            "manual_label_reason": reason,
            "manual_label_source": "codex_manual_observable-history_review_2026-08-22",
        })
        sample.append(reviewed)

    if len(sample) != 100:
        raise SystemExit(f"expected 100 review rows, found {len(sample)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    write_summary(sample, Path(args.summary))
    calibrated = write_calibrated_table(evaluation, sample, Path(args.calibrated))

    print(f"wrote {len(sample)} reviewed rows to {output}")
    print(f"manual outcomes: {dict(Counter(row['manual_outcome_label'] for row in sample))}")
    print(f"wrote {args.summary}")
    print(f"wrote {len(calibrated)} calibrated rows to {args.calibrated}")
    print(f"calibrated outcomes: {dict(Counter(row['evaluation_outcome_label'] for row in calibrated))}")


if __name__ == "__main__":
    main()
