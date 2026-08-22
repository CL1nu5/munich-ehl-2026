#!/usr/bin/env python3
"""Route trajectories with trained checkpoint + cache-aware cost report.

Usage (from repo root):
  python team/model/predict.py export/
  python team/model/predict.py export/ --adopt-frac 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "team"))
sys.path.insert(0, str(ROOT / "scripts"))

from cost_model import load_pricing, logged_route, trajectory_cost  # noqa: E402
from load_trajectories import group_trajectories, iter_requests  # noqa: E402
from model.features import features_for_request, load_feature_lookup  # noqa: E402
from model.router import RouterModel  # noqa: E402


def load_router(checkpoint: Path) -> RouterModel:
    payload = json.loads(checkpoint.read_text())
    return RouterModel.from_dict(payload["router"])


def route_trajectory(
    key: str, calls: list[dict], router: RouterModel, lookup: dict
) -> tuple[list[str], str]:
    first = calls[0]
    feat = features_for_request(first, request_id=key, lookup=lookup)
    decision = router.route(feat, logged_model=first["model"])
    return [decision["routed_model"] for _ in calls], decision.get("route_reason", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="export/ directory or trajectories_v1_01.jsonl")
    ap.add_argument("--checkpoint", default=str(ROOT / "team" / "checkpoints" / "model.json"))
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--out", default=str(ROOT / "results" / "team_routes.jsonl"))
    ap.add_argument("--adopt-frac", type=float, default=1.0, help="Adopt top fraction by cost savings")
    args = ap.parse_args()

    export_path = Path(args.export)
    router = load_router(Path(args.checkpoint))
    lookup = load_feature_lookup(args.datasets)
    pricing = load_pricing()
    adopt_frac = max(0.0, min(1.0, args.adopt_frac))

    if export_path.is_dir():
        groups = group_trajectories(r for _, chunk, r in iter_requests(export_path))
    else:
        groups = {}
        with open(export_path) as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                req = json.loads(line)
                groups[f"{export_path.name}:{i}"] = [req]

    planned = []
    for key, calls in groups.items():
        logged = logged_route(calls)
        routed, reason = route_trajectory(key, calls, router, lookup)
        c_logged, _ = trajectory_cost(calls, logged, pricing)
        c_routed, _ = trajectory_cost(calls, routed, pricing)
        planned.append(
            {
                "key": key,
                "calls": calls,
                "logged": logged,
                "routed": routed,
                "reason": reason,
                "c_logged": c_logged,
                "c_routed": c_routed,
                "savings": c_logged - c_routed,
            }
        )

    planned.sort(key=lambda r: r["savings"], reverse=True)
    n_adopt = int(round(adopt_frac * len(planned)))
    adopt_keys = {planned[i]["key"] for i in range(n_adopt)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tot_logged = tot_routed = 0.0
    with open(args.out, "w") as out:
        for rec in planned:
            logged = rec["logged"]
            if rec["key"] in adopt_keys:
                routed, c_logged, c_routed = rec["routed"], rec["c_logged"], rec["c_routed"]
                adopted, reason = True, rec["reason"]
            else:
                routed, c_logged, c_routed = logged, rec["c_logged"], rec["c_logged"]
                adopted, reason = False, "kept_logged"
            tot_logged += c_logged
            tot_routed += c_routed
            out.write(
                json.dumps(
                    {
                        "trajectory": rec["key"],
                        "n_calls": len(rec["calls"]),
                        "logged_model": logged[0],
                        "route": routed,
                        "adopted": adopted,
                        "route_reason": reason,
                        "cost_logged_usd": round(c_logged, 6),
                        "cost_routed_usd": round(c_routed, 6),
                        "switches": sum(
                            1 for i in range(1, len(routed)) if routed[i] != routed[i - 1]
                        ),
                    }
                )
                + "\n"
            )

    delta = (tot_routed / tot_logged - 1) if tot_logged else 0.0
    print(f"adopt_frac: {adopt_frac:.2f}  ({n_adopt}/{len(planned)} trajectories)")
    print(f"logged cost:  ${tot_logged:,.4f}")
    print(f"routed cost:  ${tot_routed:,.4f}  ({delta:+.1%})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
