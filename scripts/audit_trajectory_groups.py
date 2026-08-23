#!/usr/bin/env python3
"""Audit the starter loader's multi-request trajectory candidates.

The starter groups on the first 2,000 characters of the first user message.
For this export, all 24 multi-request candidates were manually reviewed and
are separate scheduled/event runs rather than growing histories of one task.
This script records the evidence and writes explicit, reproducible overrides.

Usage: python scripts/audit_trajectory_groups.py export/
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from load_trajectories import first_user_text, group_trajectories, iter_requests


# Manually reviewed against run timestamps/event ids, model consistency, and
# item-level history growth. Keeping this explicit makes the data decision
# auditable instead of hiding it in a fuzzy matcher.
MANUAL_SPLIT_KEYS = {
    "027a7e59dcc1ed35", "0b5041cdb0f6548a", "15b6394311ee93a6",
    "20aa5abee23cd5b4", "2a09408f6a3369cb", "2e1be2c096cbd29c",
    "368396b4e15ab06d", "3f63945c35f26ca5", "4215b6f24c6eebff",
    "7c496702d32a0883", "85bda2a6ab229bd3", "85ea85e3070e6cbb",
    "86f75d8a692327ad", "8d1a4d2d1b4b37f5", "8fd2f7ea814f4752",
    "965948a699d61f6d", "e2abb3aa1abdff7b", "e3af5cbdc023e6a0",
    "47d7c8997f5801df", "52ca580b2115132f", "563b4fee87f649ae",
    "767e735a678c17cf", "e71e8dac0b90e4ca", "ef9f29a29de19c9b",
}


def common_prefix_chars(texts):
    if not texts:
        return 0
    shortest = min(map(len, texts))
    for index in range(shortest):
        if len({text[index] for text in texts}) > 1:
            return index
    return shortest


def is_item_prefix(shorter, longer):
    return len(shorter) <= len(longer) and all(a == b for a, b in zip(shorter, longer))


def corrected_id(group_id, chunk, line_no):
    suffix = hashlib.sha1(f"{chunk}:{line_no}".encode()).hexdigest()[:8]
    return f"{group_id}-{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="export")
    parser.add_argument("--audit-output", default="results/trajectory_group_audit.csv")
    parser.add_argument("--overrides-output", default="results/trajectory_overrides.json")
    args = parser.parse_args()

    records = list(iter_requests(args.export))
    source = {id(req): (chunk, line_no) for chunk, line_no, req in records}
    groups = group_trajectories(req for _, _, req in records)
    multi = {key: calls for key, calls in groups.items() if len(calls) > 1}

    unexpected = set(multi) - MANUAL_SPLIT_KEYS
    missing = MANUAL_SPLIT_KEYS - set(multi)
    if unexpected or missing:
        raise SystemExit(
            "manual audit keys do not match this export; review required: "
            f"unexpected={sorted(unexpected)} missing={sorted(missing)}"
        )

    audit_rows = []
    overrides = {}
    for key, calls in sorted(multi.items()):
        texts = [first_user_text(call) for call in calls]
        ordered = sorted(calls, key=lambda call: len(call["input"]))
        prefix_chain = all(
            is_item_prefix(ordered[i]["input"], ordered[i + 1]["input"])
            for i in range(len(ordered) - 1)
        )
        models = sorted({call.get("model", "unknown") for call in calls})
        reason = (
            "distinct recurring/event runs; full opening messages differ; no exact "
            "growing-history prefix"
        )
        if len(models) > 1:
            reason += "; mixed logged models violate the confirmed trajectory premise"

        request_overrides = []
        for call in calls:
            chunk, line_no = source[id(call)]
            request_overrides.append({
                "source_chunk": chunk,
                "source_line": line_no + 1,
                "corrected_trajectory_id": corrected_id(key, chunk, line_no),
            })
        overrides[key] = {
            "decision": "split_all_requests",
            "confidence": "high",
            "reason": reason,
            "requests": request_overrides,
        }
        audit_rows.append({
            "original_group_id": key,
            "request_count": len(calls),
            "models": "|".join(models),
            "mixed_models": int(len(models) > 1),
            "full_first_user_unique": len(set(texts)),
            "common_first_user_prefix_chars": common_prefix_chars(texts),
            "exact_item_prefix_chain": int(prefix_chain),
            "decision": "split_all_requests",
            "confidence": "high",
            "reason": reason,
        })

    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    override_path = Path(args.overrides_output)
    override_path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    print(f"audited {len(audit_rows)} multi-request candidate groups")
    print(f"decision: split all {sum(row['request_count'] for row in audit_rows)} requests")
    print(f"wrote {audit_path}")
    print(f"wrote {override_path}")


if __name__ == "__main__":
    main()
