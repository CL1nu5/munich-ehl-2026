#!/usr/bin/env python3
"""Holdout evaluation + cost-quality frontier for team router.

Usage (from repo root):
  python team/eval/evaluate.py
  python team/eval/evaluate.py --routes results/team_routes.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "team"))
sys.path.insert(0, str(ROOT / "scripts"))

from cost_model import load_pricing, logged_route, trajectory_cost  # noqa: E402
from eval.quality import (  # noqa: E402
    build_match_table,
    estimate_routed_quality,
    load_models_from_export,
    load_targets,
    load_validation_targets,
    logged_outcome_quality,
)
from load_trajectories import group_trajectories, iter_requests  # noqa: E402
from model.features import features_for_request, load_feature_lookup  # noqa: E402
from model.router import RouterModel  # noqa: E402


def load_router(checkpoint: Path) -> RouterModel:
    payload = json.loads(checkpoint.read_text())
    return RouterModel.from_dict(payload["router"])


def annotate_groups(export_dir: Path) -> dict:
    annotated = []
    for chunk, line_no, req in iter_requests(export_dir):
        row = dict(req)
        row["_chunk"] = chunk
        row["_line"] = line_no + 1  # 1-based, matches validation request_id
        annotated.append(row)
    return group_trajectories(annotated)


def trajectory_for_request_id(groups: dict, request_id: str) -> tuple[str, list[dict]] | None:
    chunk, line_s = request_id.split(":", 1)
    target_line = int(line_s)
    for key, calls in groups.items():
        for c in calls:
            if c.get("_chunk") == chunk and c.get("_line") == target_line:
                return key, calls
    return None


def route_trajectory_baseline(calls: list[dict]) -> str:
    """Mirror scripts/baseline_router.py whole-trajectory policy."""
    from load_trajectories import est_tokens

    CHEAP = {"claude": "claude-sonnet-5", "gpt": "gpt-5.6-luna"}
    SMALL_TRAJECTORY = 15_000

    def cheap_for(model: str) -> str:
        return CHEAP["claude"] if model.startswith("claude") else CHEAP["gpt"]

    total = sum(est_tokens(c["input"]) for c in calls)
    if total < SMALL_TRAJECTORY:
        return cheap_for(calls[0]["model"])
    return calls[0]["model"]

def route_trajectory_team(key: str, calls: list[dict], router: RouterModel, lookup: dict) -> str:
    feat = features_for_request(calls[0], request_id=key, lookup=lookup)
    return router.route(feat, logged_model=calls[0]["model"])["routed_model"]


def eval_holdout(
    router: RouterModel,
    groups: dict,
    targets: dict[str, dict],
    lookup: dict,
    pricing: dict,
    table: dict,
) -> list[dict]:
    rows = []
    for rid, target in sorted(targets.items()):
        found = trajectory_for_request_id(groups, rid)
        if not found:
            continue
        key, calls = found
        logged = logged_route(calls)
        logged_model = logged[0]
        routed_model = route_trajectory_team(key, calls, router, lookup)
        routed = [routed_model for _ in calls]

        c_logged, _ = trajectory_cost(calls, logged, pricing)
        c_routed, _ = trajectory_cost(calls, routed, pricing)
        logged_q = logged_outcome_quality(target)
        feat = lookup.get(rid) or features_for_request(calls[0], request_id=rid, lookup=lookup)
        est_q, method = estimate_routed_quality(
            routed_model,
            logged_model,
            target["complexity_band"],
            logged_q,
            table,
            features=feat,
        )

        rows.append(
            {
                "request_id": rid,
                "trajectory_key": key,
                "n_calls": len(calls),
                "complexity_band": target["complexity_band"],
                "complexity_score": target["complexity_score"],
                "logged_model": logged_model,
                "routed_model": routed_model,
                "cost_logged_usd": round(c_logged, 6),
                "cost_routed_usd": round(c_routed, 6),
                "logged_quality": round(logged_q, 4),
                "estimated_quality": round(est_q, 4),
                "quality_method": method,
                "route_changed": routed_model != logged_model,
            }
        )
    return rows


def eval_baseline_holdout(
    groups: dict,
    targets: dict[str, dict],
    lookup: dict,
    pricing: dict,
    table: dict,
) -> list[dict]:
    rows = []
    for rid, target in sorted(targets.items()):
        found = trajectory_for_request_id(groups, rid)
        if not found:
            continue
        _key, calls = found
        logged = logged_route(calls)
        logged_model = logged[0]
        routed_model = route_trajectory_baseline(calls)
        routed = [routed_model for _ in calls]
        c_logged, _ = trajectory_cost(calls, logged, pricing)
        c_routed, _ = trajectory_cost(calls, routed, pricing)
        logged_q = logged_outcome_quality(target)
        feat = lookup.get(rid) or features_for_request(calls[0], request_id=rid, lookup=lookup)
        est_q, method = estimate_routed_quality(
            routed_model,
            logged_model,
            target["complexity_band"],
            logged_q,
            table,
            features=feat,
        )
        rows.append(
            {
                "request_id": rid,
                "logged_model": logged_model,
                "routed_model": routed_model,
                "cost_logged_usd": round(c_logged, 6),
                "cost_routed_usd": round(c_routed, 6),
                "logged_quality": round(logged_q, 4),
                "estimated_quality": round(est_q, 4),
                "quality_method": method,
                "n_calls": len(calls),
            }
        )
    return rows


def sweep_frontier(records: list[dict], quality_key: str = "estimated_quality") -> list[dict]:
    """Adopt routing for trajectories with largest cost savings first."""
    by_savings = sorted(records, key=lambda r: r["cost_logged_usd"] - r["cost_routed_usd"], reverse=True)
    total_calls = sum(r["n_calls"] for r in records)
    total_logged_cost = sum(r["cost_logged_usd"] for r in records)
    rows = []
    for frac in [i / 20 for i in range(0, 21)]:
        n_adopt = int(round(frac * len(by_savings)))
        adopt_ids = {r["request_id"] for r in by_savings[:n_adopt]}
        cost = sum(
            r["cost_routed_usd"] if r["request_id"] in adopt_ids else r["cost_logged_usd"]
            for r in records
        )
        qual = sum(
            (r[quality_key] if r["request_id"] in adopt_ids else r["logged_quality"]) * r["n_calls"]
            for r in records
        )
        rows.append(
            {
                "adopt_frac": round(frac, 3),
                "n_adopted": n_adopt,
                "cost_usd": round(cost, 4),
                "quality": round(qual / max(1, total_calls), 4),
                "cost_vs_logged_pct": round((cost / total_logged_cost - 1) * 100, 2)
                if total_logged_cost
                else 0.0,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    changed = [r for r in rows if r["route_changed"]]
    return {
        "n_holdout": n,
        "n_route_changed": len(changed),
        "frac_route_changed": round(len(changed) / n, 3),
        "total_cost_logged_usd": round(sum(r["cost_logged_usd"] for r in rows), 4),
        "total_cost_routed_usd": round(sum(r["cost_routed_usd"] for r in rows), 4),
        "cost_delta_pct": round(
            sum(r["cost_routed_usd"] for r in rows) / sum(r["cost_logged_usd"] for r in rows) - 1, 4
        )
        if sum(r["cost_logged_usd"] for r in rows)
        else 0.0,
        "mean_logged_quality": round(sum(r["logged_quality"] for r in rows) / n, 4),
        "mean_estimated_quality": round(sum(r["estimated_quality"] for r in rows) / n, 4),
        "quality_delta": round(
            sum(r["estimated_quality"] for r in rows) / n - sum(r["logged_quality"] for r in rows) / n, 4
        ),
        "exact_model_match": round(sum(r["routed_model"] == r["logged_model"] for r in rows) / n, 3),
    }


def load_routes_summary(routes_path: Path) -> dict:
    recs = [json.loads(l) for l in open(routes_path) if l.strip()]
    tot_l = sum(r["cost_logged_usd"] for r in recs)
    tot_r = sum(r["cost_routed_usd"] for r in recs)
    return {
        "n_trajectories": len(recs),
        "total_cost_logged_usd": round(tot_l, 4),
        "total_cost_routed_usd": round(tot_r, 4),
        "cost_delta_pct": round(tot_r / tot_l - 1, 4) if tot_l else 0.0,
    }


def maybe_plot_frontier(
    team_rows: list[dict], baseline_rows: list[dict], out_png: Path, title: str = "holdout"
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4.5))
        plt.plot(
            [r["cost_usd"] for r in team_rows],
            [r["quality"] for r in team_rows],
            "o-",
            color="#6748FD",
            label="team router",
        )
        plt.plot(
            [r["cost_usd"] for r in baseline_rows],
            [r["quality"] for r in baseline_rows],
            "s--",
            color="#888888",
            label="baseline",
        )
        plt.xlabel("cost (USD, est. input tokens, cache-aware)")
        plt.ylabel("estimated outcome quality (0–1)")
        plt.title(f"Cost–quality frontier ({title})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"wrote {out_png}")
    except ImportError:
        print("matplotlib not installed — skipped PNG")


def eval_complexity(router: RouterModel, targets: dict[str, dict], lookup: dict) -> dict:
    feats, labels = [], []
    for rid, t in targets.items():
        if rid not in lookup:
            continue
        feats.append(lookup[rid])
        labels.append(t)
    if not feats:
        return {}
    pred_scores, true_scores, pred_bands, true_bands = [], [], [], []
    for f, t in zip(feats, labels):
        pred = router.complexity.predict(f)
        pred_scores.append(pred["complexity_score"])
        true_scores.append(t["complexity_score"])
        pred_bands.append(pred["complexity_band"])
        true_bands.append(t["complexity_band"])
    n = len(feats)
    return {
        "n": n,
        "score_mae": round(sum(abs(a - b) for a, b in zip(true_scores, pred_scores)) / n, 3),
        "band_accuracy": round(sum(a == b for a, b in zip(true_bands, pred_bands)) / n, 3),
    }


def run_split_eval(
    split: str,
    router: RouterModel,
    groups: dict,
    targets: dict[str, dict],
    lookup: dict,
    pricing: dict,
    table: dict,
) -> dict:
    holdout_rows = eval_holdout(router, groups, targets, lookup, pricing, table)
    baseline_rows = eval_baseline_holdout(groups, targets, lookup, pricing, table)
    summary = summarize(holdout_rows)
    baseline_summary = summarize(
        [
            {
                **r,
                "route_changed": r["routed_model"] != r["logged_model"],
                "trajectory_key": "",
                "complexity_band": targets[r["request_id"]]["complexity_band"],
                "complexity_score": targets[r["request_id"]]["complexity_score"],
            }
            for r in baseline_rows
        ]
    )
    return {
        "split": split,
        "n_targets": len(targets),
        "complexity_metrics": eval_complexity(router, targets, lookup),
        "team_summary": summary,
        "baseline_summary": baseline_summary,
        "team_rows": holdout_rows,
        "baseline_rows": baseline_rows,
        "team_frontier": sweep_frontier(holdout_rows),
        "baseline_frontier": sweep_frontier(baseline_rows),
    }


def print_split_report(result: dict) -> None:
    split = result["split"]
    n = result["n_targets"]
    print(f"=== {split} evaluation ({n} samples) ===")
    cx = result.get("complexity_metrics") or {}
    if cx:
        print("  [complexity model]")
        for k, v in cx.items():
            print(f"    {k}: {v}")
    print("  [team router]")
    for k, v in result["team_summary"].items():
        print(f"    {k}: {v}")
    print("  [baseline router]")
    for k, v in result["baseline_summary"].items():
        print(f"    {k}: {v}")


def write_split_outputs(result: dict, out_dir: Path) -> None:
    split = result["split"]
    prefix = split  # validation | test

    payload = {
        "schema_version": "munich_ehl_eval_v2",
        "split": split,
        "quality_signal": "mean(observed_components); off-policy via kNN + band×model match table (fit on train)",
        "failure_modes": [
            "match table built from train split (700 labels)",
            "match cells with n<3 fall back to model or global mean",
            "routed!=logged assumes quality independent of task beyond band",
            "output tokens excluded from cost; tokens are chars/4 estimates",
        ],
        "complexity_metrics": result["complexity_metrics"],
        "team_summary": result["team_summary"],
        "baseline_summary": result["baseline_summary"],
        "team_rows": result["team_rows"],
        "frontier": result["team_frontier"],
        "baseline_frontier": result["baseline_frontier"],
    }

    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))

    frontier_csv = out_dir / f"{prefix}_frontier.csv"
    with open(frontier_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result["team_frontier"][0].keys())
        w.writeheader()
        w.writerows(result["team_frontier"])

    eval_csv = out_dir / f"{prefix}_eval.csv"
    if result["team_rows"]:
        with open(eval_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=result["team_rows"][0].keys())
            w.writeheader()
            w.writerows(result["team_rows"])

    maybe_plot_frontier(
        result["team_frontier"],
        result["baseline_frontier"],
        out_dir / f"{prefix}_frontier.png",
        title=split,
    )

    print(f"wrote {summary_path}")
    print(f"wrote {frontier_csv}")
    if result["team_rows"]:
        print(f"wrote {eval_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(ROOT / "export"))
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--checkpoint", default=str(ROOT / "team" / "checkpoints" / "model.json"))
    ap.add_argument("--routes", default=str(ROOT / "results" / "team_routes.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument(
        "--split",
        choices=("validation", "test", "both"),
        default="validation",
        help="which labeled split to evaluate (test uses match table from validation)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(args.export)
    export_file = export_dir / "trajectories_v1_01.jsonl"
    datasets_dir = Path(args.datasets)

    router = load_router(Path(args.checkpoint))
    lookup = load_feature_lookup(datasets_dir)
    train_path = datasets_dir / "train_targets.jsonl"
    quality_targets_path = train_path if train_path.exists() else datasets_dir / "validation_targets.jsonl"
    quality_targets = load_targets(quality_targets_path)
    models_by_id = load_models_from_export(export_file)
    table = build_match_table(list(quality_targets.values()), models_by_id, lookup)
    pricing = load_pricing()
    groups = annotate_groups(export_dir)

    splits = ["validation", "test"] if args.split == "both" else [args.split]
    for split in splits:
        targets_path = datasets_dir / f"{split}_targets.jsonl"
        if not targets_path.exists():
            print(f"skip {split}: missing {targets_path}")
            continue
        targets = load_targets(targets_path)
        result = run_split_eval(split, router, groups, targets, lookup, pricing, table)
        if split == "validation" and Path(args.routes).exists():
            result["team_summary"]["full_export_routes"] = load_routes_summary(Path(args.routes))
        write_split_outputs(result, out_dir)
        print_split_report(result)
        if split == "validation" and "full_export_routes" in result["team_summary"]:
            print("=== Full export (all trajectories) ===")
            for k, v in result["team_summary"]["full_export_routes"].items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
