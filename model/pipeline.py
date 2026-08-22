"""End-to-end run: dataset -> features -> heads -> held-out evaluation.

``run_pipeline`` is the single entry point the notebook and the CLI both call. It
returns a report dict and writes the same thing to ``results/complexity_model/``.

Discipline the pipeline enforces:

* Scaling, ridge penalties, MLP weights and head selection are fitted or chosen
  on train/validation only.
* The test split is scored exactly once, at the end, for the selected variant.
* The logged model id stays in the audit metadata and never becomes a feature.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

import numpy as np

from .config import TARGETS, PipelineConfig, fast_config
from .data import check_leakage, ensure_dataset, load_splits
from .evaluate import band_breakdown, confusion, regression_metrics
from .features import build_feature_matrices, target_matrix
from .heads import MeanBaseline, MLPHead, RidgeHead


@contextmanager
def _timed(store: dict, key: str):
    started = time.perf_counter()
    yield
    store[key] = round(time.perf_counter() - started, 2)


def build_heads(config: PipelineConfig) -> list:
    heads = [MeanBaseline()]
    if config.head in {"ridge", "both"}:
        heads.append(RidgeHead(config.ridge_alphas))
    if config.head in {"mlp", "both"}:
        heads.append(MLPHead(config))
    return heads


def run_pipeline(
    config: PipelineConfig | None = None,
    *,
    verbose: bool = True,
    save: bool = True,
) -> dict:
    """Run the whole workflow and return a report."""
    config = config or fast_config()
    timings: dict[str, float] = {}
    say = print if verbose else (lambda *a, **k: None)

    # 1. Dataset ---------------------------------------------------------
    with _timed(timings, "dataset"):
        manifest = ensure_dataset(config, verbose=verbose)
        splits = load_splits(config)
        integrity = check_leakage(splits, manifest)
    say(f"Dataset: {integrity['requests']} requests, "
        f"{integrity['split_groups']} prompt groups, no split leakage "
        f"({ {k: len(v) for k, v in splits.items()} })")

    # 2. Features --------------------------------------------------------
    with _timed(timings, "embedding"):
        features, embedding_stats = build_feature_matrices(splits, config, verbose=verbose)
    say(f"Features: {embedding_stats['feature_dim']} dims "
        f"({'cached' if embedding_stats.get('cached') else 'freshly encoded'})")

    targets = {name: target_matrix(split, TARGETS) for name, split in splits.items()}
    primary = TARGETS.index("complexity_score")

    # 3. Train every head; select on validation --------------------------
    comparison, fitted = [], {}
    with _timed(timings, "training"):
        for head in build_heads(config):
            head.fit(features["train"], targets["train"],
                     features["validation"], targets["validation"], primary=primary)
            prediction = head.predict(features["validation"])[:, primary]
            metrics = regression_metrics(targets["validation"][:, primary], prediction)
            fitted[head.name] = head
            comparison.append({
                "variant": head.name,
                "validation_MAE": metrics["MAE"],
                "validation_R2": metrics["R2"],
                "validation_spearman": metrics["spearman"],
                "validation_band_accuracy": metrics["band_accuracy"],
                "fit_seconds": head.describe()["fit_seconds"],
            })

    comparison.sort(key=lambda row: row["validation_MAE"])
    selected_name = comparison[0]["variant"]
    selected = fitted[selected_name]
    say(f"Selected head: {selected_name} (validation MAE {comparison[0]['validation_MAE']})")

    # 4. Score the splits; test is touched only here ---------------------
    report_splits = {}
    for name in ("train", "validation", "test"):
        prediction = selected.predict(features[name])[:, primary]
        truth = targets[name][:, primary]
        report_splits[name] = {
            "metrics": regression_metrics(truth, prediction),
            "bands": band_breakdown(truth, prediction),
            "confusion": confusion(truth, prediction),
        }
    test_metrics = report_splits["test"]["metrics"]
    say(f"Test: MAE {test_metrics['MAE']}  R2 {test_metrics['R2']}  "
        f"spearman {test_metrics['spearman']}  band acc {test_metrics['band_accuracy']}")

    report = {
        "config": config.to_json(),
        "integrity": integrity,
        "embedding": embedding_stats,
        "comparison": comparison,
        "selected": selected_name,
        "selected_detail": selected.describe(),
        "timings": timings,
        "targets": list(TARGETS),
        **report_splits,
    }

    if save:
        config.results_dir.mkdir(parents=True, exist_ok=True)
        (config.results_dir / "report.json").write_text(json.dumps(report, indent=2))
        predictions = {
            name: selected.predict(features[name])[:, primary].round(4).tolist()
            for name in splits
        }
        (config.results_dir / "predictions.json").write_text(json.dumps({
            "request_ids": {name: split.request_ids for name, split in splits.items()},
            "predicted_complexity_score": predictions,
        }))
        say(f"Wrote {config.results_dir / 'report.json'}")

    report["_features"] = features
    report["_targets"] = targets
    report["_splits"] = splits
    report["_head"] = selected
    return report


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .config import accurate_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("static", "minilm"), default="static")
    parser.add_argument("--head", choices=("ridge", "mlp", "both"), default="both")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    args = parser.parse_args(argv)

    base = fast_config() if args.encoder == "static" else accurate_config()
    run_pipeline(base.with_(
        head=args.head,
        use_cache=not args.no_cache,
        rebuild_dataset=args.rebuild_dataset,
    ))


if __name__ == "__main__":
    main()
