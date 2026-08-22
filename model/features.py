"""Assemble the predictor matrix: numeric request features + text embeddings.

Only routing-time-observable fields are used. The logged model, the targets and
every ``*_metadata.jsonl`` field are audit-only and never enter this matrix.
"""
from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from .data import Split
from .embed import embed_rows

#: The deterministic text statistics the dataset builder ships with every row.
STATIC_FEATURE_NAMES = (
    "system_est_tokens",
    "user_est_tokens",
    "user_message_count",
    "step_marker_count",
    "question_count",
    "action_term_count",
    "constraint_term_count",
    "domain_term_count",
    "image_count",
    "url_count",
    "code_block_count",
)


def static_feature_matrix(rows: list[dict]) -> np.ndarray:
    """Counts and token estimates, ``log1p``-compressed.

    These features are heavy-tailed (system prompts span 8.5k–142k characters);
    on the raw scale a handful of long prompts dominate the least-squares fit.
    """
    raw = np.array(
        [[float(row["static_text_features"].get(name, 0.0)) for name in STATIC_FEATURE_NAMES]
         for row in rows],
        dtype=np.float32,
    )
    return np.log1p(np.maximum(raw, 0.0))


class Standardizer:
    """Z-score scaler fitted on the training split only."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, matrix: np.ndarray) -> "Standardizer":
        self.mean = matrix.mean(axis=0)
        self.std = np.maximum(matrix.std(axis=0), 1e-6)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer used before fit()")
        return (matrix - self.mean) / self.std


def build_feature_matrices(
    splits: dict[str, Split],
    config: PipelineConfig,
    *,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return one predictor matrix per split, plus embedding stats.

    All rows are embedded in a single pass so that a chunk shared between splits
    is encoded once. This is not leakage: the encoder is frozen and never sees a
    target. Everything *fitted* — scaling, ridge, the MLP — uses train rows only.
    """
    if not (config.use_static_features or config.use_embeddings):
        raise ValueError("Enable at least one of use_static_features / use_embeddings")

    order = [name for name in ("train", "validation", "test") if name in splits]
    blocks: dict[str, list[np.ndarray]] = {name: [] for name in order}
    stats: dict = {}

    if config.use_embeddings:
        all_rows = [row for name in order for row in splits[name].inputs]
        matrix, stats = embed_rows(all_rows, config, tag="all", verbose=verbose)
        cursor = 0
        for name in order:
            size = len(splits[name])
            blocks[name].append(matrix[cursor : cursor + size])
            cursor += size

    if config.use_static_features:
        scaler = Standardizer().fit(static_feature_matrix(splits["train"].inputs))
        for name in order:
            blocks[name].append(scaler.transform(static_feature_matrix(splits[name].inputs)))

    features = {name: np.hstack(parts).astype(np.float32) for name, parts in blocks.items()}
    stats["feature_dim"] = int(features[order[0]].shape[1])
    return features, stats


def target_matrix(split: Split, targets: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[float(row[name]) for name in targets] for row in split.targets], dtype=np.float32
    )


def band_of(score: float | np.ndarray):
    """Dataset band rule: ``<=35`` low, ``<=70`` medium, else high.

    Verified to reproduce ``complexity_band`` on all 1000 rows of the export.
    """
    scores = np.asarray(score)
    bands = np.where(scores <= 35.0, "low", np.where(scores <= 70.0, "medium", "high"))
    return bands if bands.ndim else str(bands)
