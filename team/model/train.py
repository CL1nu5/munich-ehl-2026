#!/usr/bin/env python3
"""Train complexity + router models.

Usage (from repo root):
  python team/model/train.py
  python team/model/train.py --export export/trajectories_v1_01.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "team"))

from model.complexity import ComplexityModel  # noqa: E402
from model.features import (  # noqa: E402
    FEATURE_NAMES,
    features_for_request,
    load_validation_feature_lookup,
    vectorize,
)
from model.linear import save_checkpoint  # noqa: E402
from model.router import ROUTER_CLASSES, RouterModel  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from load_trajectories import group_trajectories, iter_requests  # noqa: E402


def load_labeled_complexity(datasets_dir: Path) -> tuple[list[dict], list[dict], list[str]]:
    inputs_path = datasets_dir / "validation_inputs.jsonl"
    targets_path = datasets_dir / "validation_targets.jsonl"
    targets = {json.loads(l)["request_id"]: json.loads(l) for l in open(targets_path)}
    features, labels, ids = [], [], []
    for line in open(inputs_path):
        row = json.loads(line)
        rid = row["request_id"]
        if rid not in targets:
            continue
        features.append(row["static_text_features"])
        labels.append(targets[rid])
        ids.append(rid)
    return features, labels, ids


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def band_acc(y_true: list[str], y_pred: list[str]) -> float:
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def load_router_training(export_path: Path, lookup: dict) -> tuple[list[dict], list[str], list[str]]:
    """First call per trajectory = routing decision point."""
    if export_path.is_dir():
        groups = group_trajectories(r for _, _, r in iter_requests(export_path))
        items = []
        for key, calls in groups.items():
            req = calls[0]
            chunk = export_path.name if export_path.is_dir() else export_path.parent.name
            # rebuild id unknown here; use group key only
            items.append((key, req, req["model"]))
        features, models, ids = [], [], []
        for key, req, model in items:
            features.append(features_for_request(req, lookup=lookup))
            models.append(model)
            ids.append(key)
        return features, models, ids

    # single file path
    groups = {}
    with open(export_path) as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            req = json.loads(line)
            rid = f"{export_path.name}:{i}"
            groups.setdefault(rid, req)
    # treat each line as its own routing point (matches validation granularity)
    features, models, ids = [], [], []
    with open(export_path) as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            req = json.loads(line)
            rid = f"{export_path.name}:{i}"
            features.append(features_for_request(req, request_id=rid, lookup=lookup))
            models.append(req["model"])
            ids.append(rid)
    return features, models, ids


def split_indices(n: int, frac: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * frac)))
    return idx[n_test:], idx[:n_test]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--export", default=str(ROOT / "export" / "trajectories_v1_01.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "team" / "checkpoints"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.2)
    args = ap.parse_args()

    datasets_dir = Path(args.datasets)
    out_dir = Path(args.out)
    lookup = load_validation_feature_lookup(datasets_dir)

    # --- Stage 1: complexity model ---
    cx_feats, cx_labels, cx_ids = load_labeled_complexity(datasets_dir)
    tr_idx, te_idx = split_indices(len(cx_feats), args.holdout, args.seed)

    cx_train_f = [cx_feats[i] for i in tr_idx]
    cx_train_y = [cx_labels[i] for i in tr_idx]
    cx_test_f = [cx_feats[i] for i in te_idx]
    cx_test_y = [cx_labels[i] for i in te_idx]

    complexity = ComplexityModel().fit(cx_train_f, cx_train_y)

    pred_scores = [complexity.predict(f)["complexity_score"] for f in cx_test_f]
    true_scores = [y["complexity_score"] for y in cx_test_y]
    pred_bands = [complexity.predict(f)["complexity_band"] for f in cx_test_f]
    true_bands = [y["complexity_band"] for y in cx_test_y]

    cx_metrics = {
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
        "score_mae": round(mae(true_scores, pred_scores), 3),
        "band_accuracy": round(band_acc(true_bands, pred_bands), 3),
    }

    # --- Stage 2: router on export (exclude validation ids from train) ---
    val_ids = set(cx_ids)
    rt_feats, rt_models, rt_ids = load_router_training(Path(args.export), lookup)
    router_train = [(f, m) for f, m, rid in zip(rt_feats, rt_models, rt_ids) if rid not in val_ids]
    router_test = [(f, m) for f, m, rid in zip(rt_feats, rt_models, rt_ids) if rid in val_ids]

    router = RouterModel(complexity=complexity)
    router.fit([f for f, _ in router_train], [m for _, m in router_train])

    rt_preds = [router.route(f, logged_model=m)["routed_model"] for f, m in router_test]
    rt_true = [m for _, m in router_test]
    rt_acc = sum(p == t for p, t in zip(rt_preds, rt_true)) / max(1, len(rt_true))

    # top-1 same-tier accuracy (group opus/sonnet/fable/gpt)
    def tier(m: str) -> str:
        if m.startswith("claude-opus"):
            return "opus"
        if m.startswith("claude-sonnet"):
            return "sonnet"
        if m.startswith("claude-fable"):
            return "fable"
        return "gpt"

    tier_acc = sum(tier(p) == tier(t) for p, t in zip(rt_preds, rt_true)) / max(1, len(rt_true))

    router_metrics = {
        "n_train": len(router_train),
        "n_test_validation_ids": len(router_test),
        "exact_accuracy": round(rt_acc, 3),
        "tier_accuracy": round(tier_acc, 3),
        "classes": ROUTER_CLASSES,
    }

    # Refit on all labeled / non-holdout data for the saved checkpoint.
    complexity_final = ComplexityModel().fit(cx_feats, cx_labels)
    router_final = RouterModel(complexity=complexity_final)
    router_final.fit([f for f, _ in router_train], [m for _, m in router_train])

    payload = {
        "schema_version": "munich_ehl_router_model_v1",
        "seed": args.seed,
        "feature_names": FEATURE_NAMES,
        "complexity_metrics": cx_metrics,
        "router_metrics": router_metrics,
        "router": router_final.to_dict(),
    }
    ckpt = out_dir / "model.json"
    save_checkpoint(ckpt, payload)

    print("=== Complexity model (holdout) ===")
    for k, v in cx_metrics.items():
        print(f"  {k}: {v}")
    print("=== Router (on validation request_ids) ===")
    for k, v in router_metrics.items():
        print(f"  {k}: {v}")
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
