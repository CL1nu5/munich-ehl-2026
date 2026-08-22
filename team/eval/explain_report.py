#!/usr/bin/env python3
"""Explainability report: per-model / per-reason cost & quality breakdown.

Usage (from repo root):
  python team/eval/explain_report.py --split both
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "team"))
sys.path.insert(0, str(ROOT / "scripts"))

from cost_model import load_pricing, logged_route, trajectory_cost  # noqa: E402
from eval.evaluate import (  # noqa: E402
    annotate_groups,
    load_router,
    trajectory_for_request_id,
)
from eval.quality import (  # noqa: E402
    build_match_table,
    estimate_routed_quality,
    load_models_from_export,
    load_targets,
    logged_outcome_quality,
)
from model.explain import explain_decision  # noqa: E402
from model.features import features_for_request, load_feature_lookup  # noqa: E402


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def build_explain_rows(
    router,
    groups: dict,
    targets: dict,
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
        feat = lookup.get(rid) or features_for_request(calls[0], request_id=rid, lookup=lookup)
        logged = logged_route(calls)
        logged_model = logged[0]
        expl = explain_decision(router, feat, logged_model=logged_model)
        routed_model = expl["routed_model"]
        routed = [routed_model for _ in calls]
        c_logged, _ = trajectory_cost(calls, logged, pricing)
        c_routed, _ = trajectory_cost(calls, routed, pricing)
        logged_q = logged_outcome_quality(target)
        est_q, q_method = estimate_routed_quality(
            routed_model,
            logged_model,
            expl.get("predicted_band", target["complexity_band"]),
            logged_q,
            table,
            features=feat,
        )
        top_feat = expl.get("feature_attribution", {}).get("top_features", [])
        rows.append(
            {
                "request_id": rid,
                "trajectory_key": key,
                "n_calls": len(calls),
                "logged_model": logged_model,
                "routed_model": routed_model,
                "classifier_model": expl.get("classifier_model"),
                "route_reason": expl.get("route_reason"),
                "predicted_band": expl.get("predicted_band"),
                "predicted_score": expl.get("predicted_score"),
                "labeled_band": target["complexity_band"],
                "labeled_score": target["complexity_score"],
                "classifier_confidence": expl.get("classifier_confidence"),
                "cost_logged_usd": round(c_logged, 6),
                "cost_routed_usd": round(c_routed, 6),
                "cost_save_usd": round(c_logged - c_routed, 6),
                "cost_delta_pct": round((c_routed / c_logged - 1) if c_logged else 0.0, 4),
                "quality_logged": round(logged_q, 4),
                "quality_estimated": round(est_q, 4),
                "quality_delta": round(est_q - logged_q, 4),
                "quality_method": q_method,
                "prior_logged": expl.get("prior_logged"),
                "prior_routed": expl.get("prior_routed"),
                "prior_delta": expl.get("prior_delta"),
                "top_feature": top_feat[0]["feature"] if top_feat else "",
                "top_feature_contrib": top_feat[0]["contribution"] if top_feat else 0.0,
                "route_changed": routed_model != logged_model,
                "transition": expl.get("transition"),
            }
        )
    return rows


def aggregate(rows: list[dict], group_key: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(group_key, ""))].append(r)
    out = []
    for key in sorted(buckets):
        items = buckets[key]
        n = len(items)
        changed = [x for x in items if x["route_changed"]]
        out.append(
            {
                group_key: key,
                "n": n,
                "frac_of_split": round(n / len(rows), 3) if rows else 0.0,
                "n_route_changed": len(changed),
                "total_cost_logged_usd": round(sum(x["cost_logged_usd"] for x in items), 4),
                "total_cost_routed_usd": round(sum(x["cost_routed_usd"] for x in items), 4),
                "cost_save_usd": round(sum(x["cost_save_usd"] for x in items), 4),
                "cost_delta_pct": round(
                    sum(x["cost_routed_usd"] for x in items) / sum(x["cost_logged_usd"] for x in items) - 1
                    if sum(x["cost_logged_usd"] for x in items)
                    else 0.0,
                    4,
                ),
                "mean_quality_logged": round(_mean([x["quality_logged"] for x in items]), 4),
                "mean_quality_estimated": round(_mean([x["quality_estimated"] for x in items]), 4),
                "quality_delta": round(
                    _mean([x["quality_estimated"] for x in items]) - _mean([x["quality_logged"] for x in items]), 4
                ),
                "mean_prior_delta": round(
                    _mean([x["prior_delta"] for x in items if x.get("prior_delta") is not None]), 4
                )
                if any(x.get("prior_delta") is not None for x in items)
                else None,
            }
        )
    return out


def transition_matrix(rows: list[dict]) -> list[dict]:
    counts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        counts[(r["logged_model"], r["routed_model"])].append(r)
    out = []
    for (logged, routed), items in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        out.append(
            {
                "logged_model": logged,
                "routed_model": routed,
                "n": len(items),
                "cost_save_usd": round(sum(x["cost_save_usd"] for x in items), 4),
                "mean_quality_delta": round(_mean([x["quality_delta"] for x in items]), 4),
                "route_changed": logged != routed,
            }
        )
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)


def print_table(title: str, records: list[dict], cols: list[str]) -> None:
    print(f"\n=== {title} ===")
    if not records:
        print("  (empty)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in records)) for c in cols}
    hdr = "  ".join(c.ljust(widths[c]) for c in cols)
    print(hdr)
    print("  ".join("-" * widths[c] for c in cols))
    for r in records:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def run_split(split: str, router, groups, targets, lookup, pricing, table, out_dir: Path) -> dict:
    rows = build_explain_rows(router, groups, targets, lookup, pricing, table)
    overall = {
        "total_cost_logged_usd": round(sum(r["cost_logged_usd"] for r in rows), 4),
        "total_cost_routed_usd": round(sum(r["cost_routed_usd"] for r in rows), 4),
        "cost_delta_pct": round(
            sum(r["cost_routed_usd"] for r in rows) / sum(r["cost_logged_usd"] for r in rows) - 1, 4
        )
        if rows and sum(r["cost_logged_usd"] for r in rows)
        else 0.0,
        "mean_quality_logged": round(_mean([r["quality_logged"] for r in rows]), 4),
        "mean_quality_estimated": round(_mean([r["quality_estimated"] for r in rows]), 4),
        "quality_delta": round(
            _mean([r["quality_estimated"] for r in rows]) - _mean([r["quality_logged"] for r in rows]), 4
        ),
        "n_route_changed": sum(1 for r in rows if r["route_changed"]),
    }
    summary = {
        "schema_version": "munich_ehl_explain_v1",
        "split": split,
        "n": len(rows),
        "overall": overall,
        "by_routed_model": aggregate(rows, "routed_model"),
        "by_logged_model": aggregate(rows, "logged_model"),
        "by_route_reason": aggregate(rows, "route_reason"),
        "by_predicted_band": aggregate(rows, "predicted_band"),
        "transitions": transition_matrix(rows),
    }

    prefix = out_dir / f"explain_{split}"
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    write_csv(out_dir / f"explain_{split}_rows.csv", rows)
    write_csv(out_dir / f"explain_{split}_by_model.csv", summary["by_routed_model"])
    write_csv(out_dir / f"explain_{split}_by_reason.csv", summary["by_route_reason"])
    write_csv(out_dir / f"explain_{split}_transitions.csv", summary["transitions"])

    print(f"\n{'='*60}\n{split.upper()} explainability report (n={len(rows)})\n{'='*60}")
    o = summary["overall"]
    print(
        f"Overall: cost {o['cost_delta_pct']:+.1%}  "
        f"quality {o.get('quality_delta', 0):+.4f}  "
        f"changed {o.get('n_route_changed', 0)}/{len(rows)}"
    )
    print_table(
        "By routed model",
        summary["by_routed_model"],
        ["routed_model", "n", "cost_delta_pct", "quality_delta", "cost_save_usd"],
    )
    print_table(
        "By route reason",
        summary["by_route_reason"],
        ["route_reason", "n", "cost_delta_pct", "quality_delta", "n_route_changed"],
    )
    print_table(
        "Top transitions (logged → routed)",
        summary["transitions"][:8],
        ["logged_model", "routed_model", "n", "cost_save_usd", "mean_quality_delta"],
    )
    print(f"\nwrote {prefix}.json and explain_{split}_*.csv")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(ROOT / "export"))
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--checkpoint", default=str(ROOT / "team" / "checkpoints" / "model.json"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--split", choices=("validation", "test", "both"), default="both")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    router = load_router(Path(args.checkpoint))
    lookup = load_feature_lookup(args.datasets)
    export_file = Path(args.export) / "trajectories_v1_01.jsonl"
    train_path = Path(args.datasets) / "train_targets.jsonl"
    quality_path = train_path if train_path.exists() else Path(args.datasets) / "validation_targets.jsonl"
    table = build_match_table(
        list(load_targets(quality_path).values()),
        load_models_from_export(export_file),
        lookup,
    )
    pricing = load_pricing()
    groups = annotate_groups(Path(args.export))

    splits = ["validation", "test"] if args.split == "both" else [args.split]
    for split in splits:
        tpath = Path(args.datasets) / f"{split}_targets.jsonl"
        if not tpath.exists():
            print(f"skip {split}: missing {tpath}")
            continue
        run_split(split, router, groups, load_targets(tpath), lookup, pricing, table, out_dir)


if __name__ == "__main__":
    main()
