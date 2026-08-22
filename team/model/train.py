#!/usr/bin/env python3
"""Train complexity + router models (v3).

Usage (from repo root):
  python team/model/train.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "team"))

from model.complexity import BAND_HIGH, BAND_LOW, ComplexityModel  # noqa: E402
from model.features import (  # noqa: E402
    FEATURE_NAMES,
    features_for_request,
    load_feature_lookup,
    load_held_out_ids,
    load_split_ids,
    load_train_ids,
)
from model.linear import save_checkpoint  # noqa: E402
from model.quality_prior import QualityPrior  # noqa: E402
from model.router import ROUTER_CLASSES, RouterModel, cost_rank  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from load_trajectories import group_trajectories, iter_requests  # noqa: E402


def load_labeled_complexity(datasets_dir: Path, split: str) -> tuple[list[dict], list[dict], list[str]]:
    inputs_path = datasets_dir / f"{split}_inputs.jsonl"
    targets_path = datasets_dir / f"{split}_targets.jsonl"
    targets = {json.loads(l)["request_id"]: json.loads(l) for l in open(targets_path) if l.strip()}
    features, labels, ids = [], [], []
    for line in open(inputs_path):
        if not line.strip():
            continue
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
    if export_path.is_dir():
        groups = group_trajectories(r for _, _, r in iter_requests(export_path))
        features, models, ids = [], [], []
        for key, calls in groups.items():
            req = calls[0]
            features.append(features_for_request(req, lookup=lookup))
            models.append(req["model"])
            ids.append(key)
        return features, models, ids

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


def eval_complexity(model: ComplexityModel, feats: list[dict], labels: list[dict], *, routing: bool = False) -> dict:
    if not feats:
        return {}
    pred_scores = [model.predict(f, routing=routing)["complexity_score"] for f in feats]
    true_scores = [y["complexity_score"] for y in labels]
    pred_bands = [model.predict(f, routing=routing)["complexity_band"] for f in feats]
    true_bands = [y["complexity_band"] for y in labels]
    return {
        "n": len(feats),
        "score_mae": round(mae(true_scores, pred_scores), 3),
        "band_accuracy": round(band_acc(true_bands, pred_bands), 3),
    }


def tune_band_thresholds(
    train_f: list[dict], train_y: list[dict], val_f: list[dict], val_y: list[dict]
) -> tuple[float, float, dict]:
    best_low, best_high = BAND_LOW, BAND_HIGH
    best_score = -1.0
    best_metrics: dict = {}
    for low in range(35, 51, 5):
        for high in range(65, 81, 5):
            if high <= low + 15:
                continue
            model = ComplexityModel(band_low=float(low), band_high=float(high)).fit(train_f, train_y)
            metrics = eval_complexity(model, val_f, val_y, routing=True)
            score = metrics["band_accuracy"] - metrics["score_mae"] / 100.0
            if score > best_score:
                best_score = score
                best_low, best_high = float(low), float(high)
                best_metrics = metrics
    return best_low, best_high, best_metrics


def tune_confidence(
    router: RouterModel, router_val: list[tuple[dict, str]], *, quality_weight: float = 8.0
) -> float:
    if not router_val:
        return router.confidence_threshold
    best_t, best_score = router.confidence_threshold, -1e18
    for i in range(5, 18):
        t = i / 20.0
        router.confidence_threshold = t
        saved_tiers = 0.0
        q_delta = 0.0
        n_changed = 0
        for f, m in router_val:
            r = router.route(f, logged_model=m)
            if r["routed_model"] != m:
                n_changed += 1
                saved_tiers += cost_rank(m) - cost_rank(r["routed_model"])
                if router.quality_prior:
                    band = r["predicted_band"]
                    q_delta += router.quality_prior.estimate(band, r["routed_model"]) - router.quality_prior.estimate(
                        band, m
                    )
        score = saved_tiers + quality_weight * q_delta + 0.1 * n_changed
        if score > best_score:
            best_score = score
            best_t = t
    router.confidence_threshold = best_t
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--export", default=str(ROOT / "export" / "trajectories_v1_01.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "team" / "checkpoints"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--complexity-split", default="train")
    ap.add_argument("--complexity-eval-split", default="validation")
    args = ap.parse_args()

    datasets_dir = Path(args.datasets)
    out_dir = Path(args.out)
    lookup = load_feature_lookup(datasets_dir)
    train_ids = load_train_ids(datasets_dir)
    held_out_ids = load_held_out_ids(datasets_dir)
    val_ids = load_split_ids(datasets_dir, "validation")

    cx_train_f, cx_train_y, _ = load_labeled_complexity(datasets_dir, args.complexity_split)
    cx_val_f, cx_val_y, _ = load_labeled_complexity(datasets_dir, args.complexity_eval_split)

    band_low, band_high, band_val_metrics = tune_band_thresholds(cx_train_f, cx_train_y, cx_val_f, cx_val_y)
    complexity_dev = ComplexityModel(band_low=band_low, band_high=band_high).fit(cx_train_f, cx_train_y)
    cx_metrics = {
        "train_split": args.complexity_split,
        "eval_split": args.complexity_eval_split,
        "n_train": len(cx_train_f),
        "band_thresholds": [band_low, band_high],
        "tuned_band_val": band_val_metrics,
        "full_score": eval_complexity(complexity_dev, cx_val_f, cx_val_y, routing=False),
        "routing_score": eval_complexity(complexity_dev, cx_val_f, cx_val_y, routing=True),
    }

    train_targets = {
        json.loads(l)["request_id"]: json.loads(l)
        for l in open(datasets_dir / "train_targets.jsonl")
        if l.strip()
    }
    export_path = Path(args.export)
    models_by_id = {}
    with open(export_path) as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                models_by_id[f"{export_path.name}:{i}"] = json.loads(line)["model"]
    quality_prior = QualityPrior.from_targets(list(train_targets.values()), models_by_id)

    rt_feats, rt_models, rt_ids = load_router_training(export_path, lookup)
    router_train_ids = train_ids - held_out_ids
    router_train = [(f, m) for f, m, rid in zip(rt_feats, rt_models, rt_ids) if rid in router_train_ids]
    router_val = [(f, m) for f, m, rid in zip(rt_feats, rt_models, rt_ids) if rid in val_ids]

    router = RouterModel(complexity=complexity_dev, quality_prior=quality_prior)
    router.fit([f for f, _ in router_train], [m for _, m in router_train])
    tuned_conf = tune_confidence(router, router_val)

    rt_preds = [router.route(f, logged_model=m)["routed_model"] for f, m in router_val]
    rt_true = [m for _, m in router_val]
    rt_acc = sum(p == t for p, t in zip(rt_preds, rt_true)) / max(1, len(rt_true))

    router_metrics = {
        "n_train": len(router_train),
        "n_train_ids_total": len(train_ids),
        "n_held_out": len(held_out_ids),
        "n_tune_validation": len(router_val),
        "confidence_threshold": round(tuned_conf, 3),
        "exact_accuracy": round(rt_acc, 3),
        "classes": ROUTER_CLASSES,
        "policy": "v3: block opus->gpt, quality veto, guarded downshift",
    }

    complexity_final = ComplexityModel(band_low=band_low, band_high=band_high).fit(cx_train_f, cx_train_y)
    router_final = RouterModel(
        complexity=complexity_final,
        confidence_threshold=tuned_conf,
        quality_prior=quality_prior,
    )
    router_final.fit([f for f, _ in router_train], [m for _, m in router_train])

    payload = {
        "schema_version": "munich_ehl_router_model_v3",
        "seed": args.seed,
        "feature_names": FEATURE_NAMES,
        "complexity_metrics": cx_metrics,
        "router_metrics": router_metrics,
        "router": router_final.to_dict(),
    }
    ckpt = out_dir / "model.json"
    save_checkpoint(ckpt, payload)

    print(f"=== Complexity v3 (bands={band_low}/{band_high}) ===")
    for k, v in cx_metrics.items():
        print(f"  {k}: {v}")
    print("=== Router v3 ===")
    for k, v in router_metrics.items():
        print(f"  {k}: {v}")
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
