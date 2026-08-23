#!/usr/bin/env python3
"""Attach conservative outcome labels to the leakage-aware routing features.

Labels describe observable trajectory history after the opening decision. The
export omits the final model output, so insufficient evidence remains UNKNOWN.
This is a telemetry proxy, not ground-truth semantic correctness.

Usage:
    python scripts/build_evaluation_table.py export/
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from build_feature_table import (
    CALL_TYPES,
    CORRECTION_RE,
    OUTPUT_TYPES,
    canonical_arguments,
    content_text,
    opening_snapshot,
    output_failed,
)
from load_trajectories import iter_requests


SIDE_EFFECT_TOOL_RE = re.compile(
    r"(?:^|_)(?:send|edit|write|delete|upload|submit|create|deploy|react|update|patch)(?:_|$)|"
    r"send_message|submit_draft|file_edit|file_write|apply_patch",
    re.I,
)
POSITIVE_RE = re.compile(
    r"\b(?:done|completed?|finished|sent|posted|delivered|uploaded|updated|created|"
    r"resolved|successful(?:ly)?|up[ -]to[ -]date)\b",
    re.I,
)
NEGATIVE_RE = re.compile(
    r"\b(?:unable to|could not|couldn't|cannot|can't|failed to|not completed|"
    r"blocked by|still blocked|unresolved|gave up)\b",
    re.I,
)
NOOP_SUCCESS_RE = re.compile(
    r"\b(?:nothing queued|queue (?:was|is) empty|no action needed|no new (?:items|"
    r"messages|replies|emails|leads)|0 (?:new|pending)|nothing to do|no writes|"
    r"already up[ -]to[ -]date)\b",
    re.I,
)


def parse_jsonish(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def output_explicit_success(output):
    value = parse_jsonish(output)
    if isinstance(value, dict):
        if value.get("success") is True or value.get("ok") is True:
            return True
        for key in ("exit_code", "exitcode", "returncode"):
            if key in value:
                try:
                    return int(value[key]) == 0
                except (TypeError, ValueError):
                    pass
        return any(
            output_explicit_success(child)
            for child in value.values()
            if isinstance(child, (dict, list))
        )
    if isinstance(value, list):
        return any(output_explicit_success(child) for child in value)
    if isinstance(value, str):
        return bool(re.search(r"(?:exit code|exit_code)\s*[:=]\s*0\b", value, re.I))
    return False


def outcome_features(request):
    items = request.get("input", [])
    opening_length = len(opening_snapshot(request).get("input", []))
    history = items[opening_length:]

    output_by_id = {
        item.get("call_id"): item.get("output")
        for item in history
        if isinstance(item, dict) and item.get("type") in OUTPUT_TYPES
    }
    calls = []
    for position, item in enumerate(history):
        if not isinstance(item, dict) or item.get("type") not in CALL_TYPES:
            continue
        output = output_by_id.get(item.get("call_id"))
        calls.append({
            "position": position,
            "name": item.get("name", "unknown"),
            "arguments": canonical_arguments(item),
            "failed": output_failed(output),
            "succeeded": output_explicit_success(output),
        })

    errors = sum(call["failed"] for call in calls)
    recovered = 0
    for index, call in enumerate(calls):
        if call["failed"] and any(
            later["name"] == call["name"] and later["succeeded"]
            for later in calls[index + 1:]
        ):
            recovered += 1
    unrecovered = max(0, errors - recovered)

    successful_side_effects = [
        call for call in calls
        if SIDE_EFFECT_TOOL_RE.search(call["name"]) and call["succeeded"] and not call["failed"]
    ]
    side_effect_pairs = Counter(
        (call["name"], call["arguments"]) for call in successful_side_effects
    )
    duplicate_side_effects = sum(count - 1 for count in side_effect_pairs.values() if count > 1)

    assistant_texts = [
        content_text(item.get("content"))
        for item in history
        if isinstance(item, dict) and item.get("role") == "assistant"
    ]
    later_user_text = "\n".join(
        content_text(item.get("content"))
        for item in history
        if isinstance(item, dict) and item.get("role") == "user"
    )
    output_text = "\n".join(
        str(item.get("output", ""))
        for item in history
        if isinstance(item, dict) and item.get("type") in OUTPUT_TYPES
    )
    final_assistant = next((text for text in reversed(assistant_texts) if text.strip()), "")

    correction = bool(CORRECTION_RE.search(later_user_text))
    positive = bool(POSITIVE_RE.search(final_assistant))
    negative = bool(NEGATIVE_RE.search(final_assistant))
    noop_success = bool(NOOP_SUCCESS_RE.search(final_assistant + "\n" + output_text))
    has_completion_evidence = bool(successful_side_effects) or positive or noop_success

    if correction and (negative or not has_completion_evidence):
        label, score, confidence = "FAILURE", 0.0, "high"
        reason = "explicit later user correction without clear subsequent completion"
    elif negative and not has_completion_evidence:
        label, score, confidence = "FAILURE", 0.0, "medium"
        reason = "explicit negative completion language"
    elif errors and recovered and has_completion_evidence:
        label, score, confidence = "RECOVERED", 0.7, "medium"
        reason = "tool error followed by same-tool success and completion evidence"
    elif successful_side_effects and not correction and not negative:
        label, score, confidence = "SUCCESS", 1.0, "high"
        reason = "structured successful side effect with no correction or negative completion"
    elif noop_success and not negative:
        label, score, confidence = "SUCCESS", 1.0, "medium"
        reason = "expected no-op completion signal"
    elif positive and not negative:
        label, score, confidence = "SUCCESS", 1.0, "medium"
        reason = "positive completion language in observable history"
    else:
        label, score, confidence = "UNKNOWN", None, "low"
        reason = "insufficient observable evidence; final model output is absent"

    return {
        "y_outcome_label": label,
        "y_outcome_score": "" if score is None else score,
        "y_label_confidence": confidence,
        "y_label_reason": reason,
        "y_label_scope": "observable_history_before_missing_final_output",
        "y_final_output_missing": 1,
        "y_observed_history_items": len(history),
        "y_observed_tool_calls": len(calls),
        "y_tool_errors": errors,
        "y_recovered_tool_errors": recovered,
        "y_unrecovered_tool_errors": unrecovered,
        "y_successful_side_effects": len(successful_side_effects),
        "y_duplicate_successful_side_effects": duplicate_side_effects,
        "y_user_correction": int(correction),
        "y_positive_completion": int(positive),
        "y_negative_completion": int(negative),
        "y_noop_success": int(noop_success),
        "y_primary_eval_eligible": int(label != "UNKNOWN" and confidence == "high"),
        "y_sensitivity_eval_eligible": int(label != "UNKNOWN" and confidence in {"high", "medium"}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="export")
    parser.add_argument("--features", default="results/trajectory_features.csv")
    parser.add_argument("--output", default="results/evaluation_table.csv")
    args = parser.parse_args()

    requests = {
        (chunk, line_no + 1): request
        for chunk, line_no, request in iter_requests(args.export)
    }
    with open(args.features, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        key = (row["source_chunk"], int(row["source_line"]))
        row.update(outcome_features(requests[key]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = Counter(row["y_outcome_label"] for row in rows)
    confidence = Counter(row["y_label_confidence"] for row in rows)
    print(f"wrote {len(rows)} rows to {output}")
    print(f"outcomes: {dict(labels)}")
    print(f"confidence: {dict(confidence)}")
    print(f"primary-eval eligible: {sum(int(row['y_primary_eval_eligible']) for row in rows)}")
    print(f"sensitivity-eval eligible: {sum(int(row['y_sensitivity_eval_eligible']) for row in rows)}")
    print("UNKNOWN rows remain excluded; the export omits every final model output.")


if __name__ == "__main__":
    main()
