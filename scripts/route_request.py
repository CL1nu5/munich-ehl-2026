#!/usr/bin/env python3
"""Route an opening request locally, without executing it or calling an LLM.

Usage: python3 scripts/route_request.py examples/request.json
"""

import argparse
import json
from pathlib import Path

from build_feature_table import extract_features, opening_snapshot, sanitize_routing_text
from train_two_stage_router import choose_action, estimated_opening_cost, predict_execution_risk


ROOT = Path(__file__).resolve().parents[1]


def route_request(request, artifact):
    features, text = extract_features(opening_snapshot(request))
    # The saved policy consumes the same string-valued rows as csv.DictReader.
    features = {name: str(value) for name, value in features.items()}
    features["x_task_text_sanitized"] = sanitize_routing_text(text)
    action, workload, quality, support, reason = choose_action(
        features,
        artifact,
        artifact["quality_tolerance"],
        artifact["minimum_exact_stratum"],
    )
    return {
        "model": artifact["profiles"][action]["route_model"],
        "reason": reason,
        "supported_actions": [name for name, result in support.items() if result[0]],
        "predicted_workload": round(workload, 4),
        "predicted_execution_risk": round(
            predict_execution_risk(features, artifact["complexity_model"]), 4
        ),
        "predicted_quality_proxy": round(quality[action], 4),
        "opening_input_cost_usd_est": round(
            estimated_opening_cost(features, action, artifact), 6
        ),
        "note": "Predictions only. Prices are assumed and tokens estimated. No request was executed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="Responses-format request JSON")
    parser.add_argument(
        "--artifact", type=Path, default=ROOT / "models/two_stage_router.json"
    )
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not isinstance(request, dict) or not isinstance(request.get("input"), list):
        parser.error("request must be a JSON object with an 'input' array")
    if not isinstance(request.get("tools", []), list):
        parser.error("request 'tools' must be an array")
    print(json.dumps(route_request(request, artifact), indent=2))


if __name__ == "__main__":
    main()
