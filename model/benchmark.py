"""Compare feature/encoder variants on wall clock and validation quality.

Every variant is scored on **validation only**. The test split stays sealed until
:func:`model.pipeline.run_pipeline` scores the one selected configuration, so the
comparison below cannot quietly turn into test-set tuning.
"""
from __future__ import annotations

import time

from .config import PipelineConfig, accurate_config, fast_config
from .evaluate import regression_metrics
from .pipeline import run_pipeline


def default_variants() -> dict[str, PipelineConfig]:
    """The four decisions worth measuring before committing to a configuration."""
    return {
        "features only": fast_config(use_embeddings=False),
        "static embeddings only": fast_config(use_static_features=False),
        "static + features": fast_config(),
        "minilm + features": accurate_config(),
    }


def run_variant(name: str, config: PipelineConfig, *, verbose: bool = False) -> dict:
    started = time.perf_counter()
    report = run_pipeline(config, verbose=verbose, save=False)
    best = report["comparison"][0]
    return {
        "variant": name,
        "encoder": config.encoder if config.use_embeddings else "—",
        "dims": report["embedding"]["feature_dim"],
        "head": best["variant"],
        "val_MAE": best["validation_MAE"],
        "val_R2": best["validation_R2"],
        "val_spearman": best["validation_spearman"],
        "val_band_acc": best["validation_band_accuracy"],
        "cold_encode_s": report["embedding"].get("cold_encode_seconds", 0.0),
        "chunks_per_s": report["embedding"].get("chunks_per_second", 0),
        "train_s": report["timings"].get("training", 0.0),
        "total_s": round(time.perf_counter() - started, 2),
        "encoded_chunks": report["embedding"].get("chunks_encoded", 0),
        "cached": bool(report["embedding"].get("cached", False)),
    }


def compare(
    variants: dict[str, PipelineConfig] | None = None,
    *,
    verbose: bool = True,
) -> list[dict]:
    variants = variants if variants is not None else default_variants()
    rows = []
    for name, config in variants.items():
        if verbose:
            print(f"--- {name}")
        row = run_variant(name, config, verbose=False)
        rows.append(row)
        if verbose:
            print(f"    val MAE {row['val_MAE']}  R2 {row['val_R2']}  "
                  f"spearman {row['val_spearman']}  |  cold encode {row['cold_encode_s']}s "
                  f"train {row['train_s']}s{' (embeddings served from cache)' if row['cached'] else ''}")
    return rows


if __name__ == "__main__":
    from .evaluate import format_metrics_table

    rows = compare()
    print()
    print(format_metrics_table(rows, (
        "variant", "encoder", "dims", "head", "val_MAE", "val_R2",
        "val_spearman", "val_band_acc", "encoded_chunks", "chunks_per_s",
        "cold_encode_s", "train_s",
    )))
