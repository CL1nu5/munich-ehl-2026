#!/usr/bin/env python3
"""Route trajectories with trained checkpoint + cache-aware cost report.

Usage (from repo root):
  python team/model/predict.py export/
  python team/model/predict.py export/trajectories_v1_01.jsonl --checkpoint team/checkpoints/model.json
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
) -> list[str]:
    """Pick model from first call's static features; apply to whole trajectory."""
    first = calls[0]
    request_id = first.get("_request_id", key)
    feat = features_for_request(first, request_id=request_id, lookup=lookup)
    chosen = router.route(feat, logged_model=first["model"])["routed_model"]
    return [chosen for _ in calls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="export/ directory or trajectories_v1_01.jsonl")
    ap.add_argument("--checkpoint", default=str(ROOT / "team" / "checkpoints" / "model.json"))
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--out", default=str(ROOT / "results" / "team_routes.jsonl"))
    args = ap.parse_args()

    export_path = Path(args.export)
    router = load_router(Path(args.checkpoint))
    lookup = load_feature_lookup(args.datasets)
    pricing = load_pricing()

    if export_path.is_dir():
        annotated = []
        for chunk, line_no, req in iter_requests(export_path):
            row = dict(req)
            row["_request_id"] = f"{chunk}:{line_no + 1}"
            annotated.append(row)
        groups = group_trajectories(annotated)
    else:
        groups = {}
        with open(export_path) as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                req = json.loads(line)
                rid = f"{export_path.name}:{i}"
                groups[rid] = [req]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tot_logged = tot_routed = 0.0
    with open(args.out, "w") as out:
        for key, calls in groups.items():
            logged = logged_route(calls)
            routed = route_trajectory(key, calls, router, lookup)
            c_logged, _ = trajectory_cost(calls, logged, pricing)
            c_routed, _ = trajectory_cost(calls, routed, pricing)
            tot_logged += c_logged
            tot_routed += c_routed
            out.write(
                json.dumps(
                    {
                        "trajectory": key,
                        "n_calls": len(calls),
                        "logged_model": logged[0],
                        "route": routed,
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
    print(f"logged cost:  ${tot_logged:,.4f}")
    print(f"routed cost:  ${tot_routed:,.4f}  ({delta:+.1%})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
