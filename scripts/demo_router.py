#!/usr/bin/env python3
"""Replay two real held-out routing decisions for the five-minute demo.

This is deliberately offline and deterministic: it reads the final held-out
decision table rather than calling an LLM. It demonstrates both sides of the
policy—saving cost when predicted quality is close, and spending more when the
quality guardrail rejects cheaper candidates.

Usage:
    python scripts/demo_router.py
    python scripts/demo_router.py --pause
"""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {
        "trajectory_id": "18273dd7606d07c7",
        "title": "CASE 1 · SAVE WHEN QUALITY IS CLOSE",
        "task": "Recurring scheduled status update",
        "interpretation": "The quality tolerance admits Sonnet, so price decides.",
    },
    {
        "trajectory_id": "a40372dd055cf212",
        "title": "CASE 2 · SPEND WHEN THE GUARDRAIL FIRES",
        "task": "Multi-source research artifact",
        "interpretation": "The cheaper route is rejected, so the policy moves up to Opus.",
    },
]


def read_by_id(path):
    if not path.is_file():
        raise SystemExit(
            f"Missing {path.name}. Run 'make pipeline' with the challenge data first, "
            "or 'make demo' for the synthetic example."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["trajectory_id"]: row for row in csv.DictReader(handle)}


def money_change(before, after):
    if before == 0:
        return 0.0
    return (after / before - 1.0) * 100.0


def print_case(case, decision, feature):
    before = float(decision["incumbent_opening_cost_usd_est"])
    after = float(decision["policy_opening_cost_usd_est"])
    change = money_change(before, after)
    sign = "+" if change >= 0 else ""
    print("\n" + "═" * 72)
    print(case["title"])
    print("═" * 72)
    print(f"Task                 {case['task']}")
    print(f"Category             {feature['x_task_category']}")
    print(f"Predicted workload   {float(decision['predicted_complexity']):5.1f} / 100")
    print(f"Predicted risk       {float(decision['predicted_execution_risk']):5.1%}")
    print()
    print(f"Logged model         {decision['logged_model']}")
    print(f"Router choice        {decision['chosen_model']}")
    print(f"Assumed input cost   ${before:.4f} → ${after:.4f}  ({sign}{change:.1f}%)")
    print(f"Predicted quality    {float(decision['predicted_quality']):.3f}")
    print()
    print("WHY  " + case["interpretation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", action="store_true", help="pause between the two cases")
    args = parser.parse_args()

    decisions = read_by_id(ROOT / "results" / "two_stage_test.csv")
    features = read_by_id(ROOT / "results" / "trajectory_features.csv")

    print("VIKTOR ROUTER · HELD-OUT DECISION REPLAY")
    print("No API calls. Values come from the untouched test-set evaluation.")
    for index, case in enumerate(CASES):
        trajectory_id = case["trajectory_id"]
        if trajectory_id not in decisions or trajectory_id not in features:
            raise SystemExit(
                f"Original replay row missing: {trajectory_id}. The current export or "
                "test split differs from the saved demo. Use 'make demo' for the "
                "synthetic example."
            )
        print_case(case, decisions[trajectory_id], features[trajectory_id])
        if args.pause and index == 0:
            input("\nPress Enter for the guardrail case…")

    print("\nTakeaway: quality constrains the route; cost chooses within that constraint.")


if __name__ == "__main__":
    main()
