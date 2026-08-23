#!/usr/bin/env python3
"""Build a leakage-aware routing feature table from the trajectory export.

The default output has one row per reconstructed trajectory and uses only the
earliest observed call in that trajectory. `--scope call` instead emits one
row per LLM request, using only the input snapshot available before that call.

The `x_*` columns are routing inputs. Audit columns such as `logged_model`,
`observed_calls`, and `grouping_status` are deliberately not prefixed with
`x_`; do not pass them to a router as features.

Usage:
    python scripts/build_feature_table.py export/
    python scripts/build_feature_table.py export/ --scope call
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from load_trajectories import est_tokens, group_trajectories, iter_requests


CALL_TYPES = {"function_call", "custom_tool_call"}
OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output"}

SCHEDULED_RE = re.compile(r"\b(?:cron|scheduled|scheduler|automation|pre-run)\b", re.I)
COMMUNICATION_RE = re.compile(
    r"\b(?:send|post|reply|respond|message|email|slack|msteams|teams|notify|dm)\b",
    re.I,
)
CODING_RE = re.compile(
    r"\b(?:code|script|python|javascript|typescript|repository|repo|git|test|"
    r"debug|deploy|api|function|database|sql)\b",
    re.I,
)
ARTIFACT_RE = re.compile(
    r"\b(?:document|spreadsheet|workbook|presentation|slides?|pdf|file|report|"
    r"image|audio|video|transcrib|dashboard)\b",
    re.I,
)
RESEARCH_RE = re.compile(
    r"\b(?:find|search|research|investigate|look up|verify|compare|analyse|analyze|"
    r"summarize|review|audit|check)\b",
    re.I,
)
CORRECTION_RE = re.compile(
    r"(?:\b(?:this|that|it) (?:is|was|'s) (?:wrong|incorrect|confusing|not what)\b|"
    r"\byou (?:missed|forgot|misunderstood|didn't|did not)\b|"
    r"\bplease (?:redo|fix|correct|revise|try again)\b|"
    r"\bnot what (?:i|we) (?:asked|wanted|meant)\b)",
    re.I,
)
MULTISTEP_RE = re.compile(
    r"\b(?:first|second|then|after that|before you|step \d+|finally|once .* then)\b",
    re.I,
)
EXTERNAL_ACTION_RE = re.compile(
    r"\b(?:send|post|reply|email|notify|upload|submit|publish|deploy|update|edit|"
    r"create|write|delete|remove|react)\b",
    re.I,
)
HIGH_RISK_RE = re.compile(
    r"\b(?:delete|remove permanently|production|deploy|payment|invoice|financial|"
    r"legal|contract|tax|irs|credential|password|secret|medical|privacy)\b",
    re.I,
)

TOOL_CATEGORIES = {
    "communication": re.compile(r"slack|msteams|teams|email|gmail|outlook|message|dm", re.I),
    "file": re.compile(r"file_|upload|download|document|spreadsheet|pdf", re.I),
    "code": re.compile(r"bash|shell|apply_patch|git|github|code|deploy", re.I),
    "research": re.compile(r"search|browser|web|history|lookup|fetch|query", re.I),
    "media": re.compile(r"image|audio|video|speech|transcri|render", re.I),
    "write": re.compile(r"send|edit|write|delete|upload|submit|create|deploy|react|update|patch", re.I),
}


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text") is not None
        )
    return ""


def user_text(input_items):
    return "\n".join(
        content_text(item.get("content"))
        for item in input_items
        if isinstance(item, dict) and item.get("role") == "user"
    )


def routing_text(input_items):
    """Favor the tail, where event triggers and concrete instructions usually live."""
    text = user_text(input_items)
    text = re.sub(r"\[base64 image redacted\]", " ", text, flags=re.I)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[-6000:]


def sanitize_routing_text(text):
    """Remove identifiers and recurring delivery boilerplate before text modeling.

    The resulting text is safe to use as an opening-time routing feature.  Generic
    placeholders are retained where their presence is informative, while their
    values are discarded so the model cannot memorize people, channels, dates,
    paths, or task ids.
    """
    text = re.sub(r"<delivery_note>.*?</delivery_note>", " ", text, flags=re.I | re.S)
    text = re.sub(r"\[base64 image redacted\]", " image ", text, flags=re.I)
    text = re.sub(r"`[^`]*`", " path ", text)
    text = re.sub(r"https?://\S+|<mailto:[^>]+>|\b[\w.+-]+@[\w.-]+\b", " url ", text, flags=re.I)
    text = re.sub(r"\bPII_[A-Z0-9_]+\b", " entity ", text, flags=re.I)
    text = re.sub(r"<[A-Z][A-Z0-9_]*(?:_[A-Z0-9_]+)*>", " entity ", text)
    text = re.sub(r"\b(?:[A-Z][A-Z0-9]{7,}|[a-f0-9]{12,})\b", " id ", text, flags=re.I)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b", " date ", text)
    text = re.sub(r"\b\d+(?:\.\d+)*\b", " number ", text)
    text = re.sub(r"\s*-?\s*cron path\s*:.*?(?:</system>|$)", " ", text, flags=re.I | re.S)
    text = re.sub(r"</?[a-z_][^>]*>", " ", text, flags=re.I)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tool_names(tools):
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
        if name:
            names.append(str(name))
    return names


def output_failed(output):
    if output is None:
        return False
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            return bool(
                re.search(r"(?:exit code|exit_code)\s*[:=]\s*[1-9]\d*\b", output, re.I)
                or re.search(r"traceback \(most recent call last\)", output, re.I)
            )
    if isinstance(output, list):
        return any(output_failed(value) for value in output)
    if not isinstance(output, dict):
        return False
    if output.get("success") is False or output.get("ok") is False:
        return True
    if output.get("is_error") is True or output.get("isError") is True:
        return True
    for key in ("exit_code", "exitcode", "returncode"):
        if key in output:
            try:
                if int(output[key]) != 0:
                    return True
            except (TypeError, ValueError):
                pass
    if output.get("error") not in (None, "", False, [], {}):
        return True
    return any(output_failed(value) for value in output.values() if isinstance(value, (dict, list)))


def canonical_arguments(item):
    raw = item.get("arguments", item.get("input", ""))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            pass
    return json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def prior_execution_features(input_items):
    outputs = {
        item.get("call_id"): item.get("output")
        for item in input_items
        if isinstance(item, dict) and item.get("type") in OUTPUT_TYPES
    }
    calls = [
        item for item in input_items
        if isinstance(item, dict) and item.get("type") in CALL_TYPES
    ]
    pairs = Counter((item.get("name", "unknown"), canonical_arguments(item)) for item in calls)
    return {
        "x_prior_tool_calls": len(calls),
        "x_prior_unique_tools": len({item.get("name", "unknown") for item in calls}),
        "x_prior_tool_errors": sum(output_failed(outputs.get(item.get("call_id"))) for item in calls),
        "x_prior_exact_duplicate_calls": sum(count - 1 for count in pairs.values() if count > 1),
    }


def count_content_parts(input_items, part_type):
    total = 0
    for item in input_items:
        if not isinstance(item, dict) or not isinstance(item.get("content"), list):
            continue
        total += sum(
            isinstance(part, dict) and part.get("type") == part_type
            for part in item["content"]
        )
    return total


def primary_category(flags):
    for name in ("scheduled", "coding", "artifact", "research", "communication"):
        if flags[name]:
            return name
    return "general"


def extract_features(request):
    items = request.get("input", [])
    tools = request.get("tools", [])
    text = routing_text(items)
    names = tool_names(tools)

    flags = {
        "scheduled": bool(SCHEDULED_RE.search(text)),
        "communication": bool(COMMUNICATION_RE.search(text)),
        "coding": bool(CODING_RE.search(text)),
        "artifact": bool(ARTIFACT_RE.search(text)),
        "research": bool(RESEARCH_RE.search(text)),
    }
    external_action = bool(EXTERNAL_ACTION_RE.search(text))
    high_risk = bool(HIGH_RISK_RE.search(text))
    if high_risk:
        risk_level = 3
    elif external_action:
        risk_level = 2
    elif flags["coding"] or flags["artifact"]:
        risk_level = 1
    else:
        risk_level = 0

    role_counts = Counter(
        item.get("role") for item in items
        if isinstance(item, dict) and item.get("role") is not None
    )
    image_parts = count_content_parts(items, "input_image")
    features = {
        "x_input_items": len(items),
        "x_input_tokens_est": est_tokens(items),
        "x_tool_schema_tokens_est": est_tokens(tools),
        "x_total_context_tokens_est": est_tokens(items) + est_tokens(tools),
        "x_system_messages": role_counts["system"],
        "x_user_messages": role_counts["user"],
        "x_assistant_messages": role_counts["assistant"],
        "x_image_parts": image_parts,
        "x_has_image": int(image_parts > 0),
        "x_available_tools": len(names),
        "x_task_text_chars": len(text),
        "x_task_text_tokens_est": max(1, len(text) // 4) if text else 0,
        "x_task_word_count": len(re.findall(r"\b\w+\b", text)),
        "x_question_marks": text.count("?"),
        "x_instruction_markers": len(re.findall(r"(?:^|\s)(?:\d+[.)]|[-*])\s+", text)),
        "x_is_scheduled": int(flags["scheduled"]),
        "x_is_communication": int(flags["communication"]),
        "x_is_coding": int(flags["coding"]),
        "x_is_artifact": int(flags["artifact"]),
        "x_is_research": int(flags["research"]),
        "x_is_correction": int(bool(CORRECTION_RE.search(text))),
        "x_requires_multiple_steps": int(bool(MULTISTEP_RE.search(text))),
        "x_requires_external_action": int(external_action),
        "x_high_risk_language": int(high_risk),
        "x_risk_level": risk_level,
        "x_task_category": primary_category(flags),
    }
    for category, pattern in TOOL_CATEGORIES.items():
        features[f"x_available_{category}_tools"] = sum(bool(pattern.search(name)) for name in names)
    features.update(prior_execution_features(items))
    return features, text


def opening_snapshot(request):
    """Return only the context available at the trajectory's opening decision."""
    opening = []
    for item in request.get("input", []):
        if not isinstance(item, dict):
            break
        if item.get("type") in CALL_TYPES or item.get("type") == "reasoning":
            break
        if item.get("role") == "assistant":
            break
        opening.append(item)
    copy = dict(request)
    copy["input"] = opening
    return copy


def grouping_status(calls):
    if len(calls) == 1:
        return "singleton"
    if len({call.get("model", "unknown") for call in calls}) > 1:
        return "mixed_model_conflict"
    return "provisional_multi_same_model"


def load_overrides(path):
    override_path = Path(path)
    if not override_path.exists():
        return {}
    return json.loads(override_path.read_text(encoding="utf-8"))


def corrected_groups(records, overrides):
    source_by_object = {
        id(request): (chunk, line_no) for chunk, line_no, request in records
    }
    original = group_trajectories(request for _, _, request in records)
    corrected = []
    for original_id, calls in sorted(original.items()):
        override = overrides.get(original_id)
        if not override:
            corrected.append((original_id, original_id, calls, "singleton" if len(calls) == 1 else grouping_status(calls)))
            continue
        if override.get("decision") != "split_all_requests":
            raise ValueError(f"unsupported override for {original_id}: {override}")
        id_by_source = {
            (row["source_chunk"], row["source_line"] - 1): row["corrected_trajectory_id"]
            for row in override["requests"]
        }
        for call in calls:
            source_key = source_by_object[id(call)]
            corrected_id = id_by_source[source_key]
            corrected.append((corrected_id, original_id, [call], "audited_split"))
    return corrected, source_by_object


def build_rows(export_dir, scope, overrides_path):
    records = list(iter_requests(export_dir))
    groups, source_by_object = corrected_groups(records, load_overrides(overrides_path))
    rows = []
    for trajectory_id, original_group_id, calls, status in sorted(groups):
        models = sorted({call.get("model", "unknown") for call in calls})
        selected = calls if scope == "call" else calls[:1]
        for call_index, request in enumerate(selected):
            chunk, line_no = source_by_object[id(request)]
            feature_request = request if scope == "call" else opening_snapshot(request)
            features, text = extract_features(feature_request)
            sanitized_text = sanitize_routing_text(text)
            row = {
                "trajectory_id": trajectory_id,
                "original_group_id": original_group_id,
                "call_index": call_index if scope == "call" else 0,
                "source_chunk": chunk,
                "source_line": line_no + 1,
                "logged_model": request.get("model", "unknown"),
                "logged_models_in_group": "|".join(models),
                "observed_calls": len(calls),
                "grouping_status": status,
                "eligible_for_modeling": int(status in {"singleton", "audited_split"}),
                "needs_manual_group_review": int(status not in {"singleton", "audited_split"}),
                "task_preview": text[-500:],
                "x_task_text_sanitized": sanitized_text,
            }
            row.update(features)
            rows.append(row)
    return rows


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="export")
    parser.add_argument("--scope", choices=("trajectory", "call"), default="trajectory")
    parser.add_argument("--output", default="results/trajectory_features.csv")
    parser.add_argument("--overrides", default="results/trajectory_overrides.json")
    args = parser.parse_args()

    rows = build_rows(args.export, args.scope, args.overrides)
    if not rows:
        raise SystemExit("no feature rows produced")
    output = Path(args.output)
    write_csv(rows, output)

    status_counts = Counter(row["grouping_status"] for row in rows)
    eligible = sum(row["eligible_for_modeling"] for row in rows)
    print(f"wrote {len(rows)} {args.scope}-level rows to {output}")
    print(f"grouping status: {dict(status_counts)}")
    print(f"eligible without manual grouping review: {eligible}/{len(rows)}")
    print("Use only x_* columns as router features; logged_model is the observed treatment.")
    print("All token counts are estimates (serialized characters / 4).")


if __name__ == "__main__":
    main()
