#!/usr/bin/env python3
"""Apply the trained two-stage router to opening-context feature rows.

Usage:
    python scripts/two_stage_router.py

Reads `results/trajectory_features.csv` and `models/two_stage_router.json`, then
writes one route per corrected trajectory to `results/two_stage_routes.jsonl`.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from train_two_stage_router import choose_action, predict_execution_risk, predict_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="results/trajectory_features.csv")
    parser.add_argument("--artifact", default="models/two_stage_router.json")
    parser.add_argument("--output", default="results/two_stage_routes.jsonl")
    args = parser.parse_args()

    with Path(args.features).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    tolerance = artifact["quality_tolerance"]
    min_stratum = artifact["minimum_exact_stratum"]

    routes = []
    for row in rows:
        action, complexity, quality, support, reason = choose_action(
            row, artifact, tolerance, min_stratum
        )
        model = artifact["profiles"][action]["route_model"]
        n_calls = int(row.get("observed_calls") or 1)
        incumbent_model = row.get("logged_model")
        routes.append({
            "trajectory": row["trajectory_id"],
            "n_calls": n_calls,
            "incumbent_model": incumbent_model,
            "route": [model] * n_calls,
            "predicted_complexity": round(complexity, 6),
            "predicted_execution_risk": round(
                predict_execution_risk(row, artifact["complexity_model"]), 6
            ),
            "predicted_quality": round(
                quality.get(action, predict_quality(row, complexity, action, artifact)), 6
            ),
            "routing_action": reason,
            "supported_actions": [
                candidate for candidate, result in support.items() if result[0]
            ],
            "benchmark_mapping_confidence": artifact["profiles"][action]["mapping_confidence"],
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for route in routes:
            handle.write(json.dumps(route) + "\n")

    counts = Counter(route["route"][0] for route in routes)
    comparable = [route for route in routes if route["incumbent_model"]]
    changed = sum(
        route["route"][0] != route["incumbent_model"] for route in comparable
    )
    print(f"wrote {len(routes)} routes to {output}")
    print(f"changed={changed}/{len(comparable)} comparable; routed models={dict(counts)}")


if __name__ == "__main__":
    main()
