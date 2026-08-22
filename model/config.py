"""Configuration for the end-to-end complexity-model pipeline.

Every knob that changes a produced artefact lives here, so a run is reproducible
from a single object and caches can be keyed on it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Sentence-transformer backends. ``static`` is a token-embedding lookup with mean
#: pooling (no transformer forward pass); ``minilm`` is a 6-layer transformer.
ENCODERS = {
    "static": "sentence-transformers/static-retrieval-mrl-en-v1",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

#: Text segments embedded separately. Keeping them apart lets the head weight the
#: near-boilerplate system prompt differently from the request-specific user text.
SEGMENTS = ("system", "user")

#: Regression targets. ``complexity_score`` is the headline; the other two are
#: auxiliary supervision (the score is a weighted blend of them).
TARGETS = ("complexity_score", "intrinsic_complexity", "observed_difficulty")


@dataclass(frozen=True)
class PipelineConfig:
    """Fully determines a pipeline run."""

    # --- paths -------------------------------------------------------------
    root: Path = ROOT
    export_dir: Path = ROOT / "export"
    dataset_dir: Path = ROOT / "results" / "complexity_dataset"
    cache_dir: Path = ROOT / "results" / "embedding_cache"
    results_dir: Path = ROOT / "results" / "complexity_model"

    # --- dataset build -----------------------------------------------------
    seed: int = 42
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    rebuild_dataset: bool = False

    # --- embedding ---------------------------------------------------------
    encoder: str = "static"
    #: Chunk size in characters. ~4 chars/token, so 960 ≈ 240 tokens, which fits
    #: MiniLM's 256-token window with room for the special tokens.
    max_chunk_chars: int = 960
    #: Per-segment chunk budgets; ``0`` means uncapped. The system prompt is mostly
    #: shared boilerplate whose length is already a numeric feature, so it gets the
    #: tighter budget. A sweep from 4/16 to 64/256 moved validation MAE by <0.09,
    #: so these sit at the cheap end of a plateau rather than at a tuned optimum.
    system_max_chunks: int = 8
    user_max_chunks: int = 32
    batch_size: int = 256
    device: str | None = None  # None -> auto-detect
    use_fp16: bool = True
    use_cache: bool = True

    # --- head --------------------------------------------------------------
    head: str = "both"  # "ridge" | "mlp" | "both"
    use_static_features: bool = True
    use_embeddings: bool = True
    ridge_alphas: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    mlp_hidden: int = 128
    mlp_dropout: float = 0.15
    mlp_epochs: int = 400
    mlp_patience: int = 60
    mlp_lr: float = 3e-3
    mlp_weight_decay: float = 1e-4
    #: Loss weights for (complexity_score, intrinsic_complexity, observed_difficulty).
    target_weights: tuple[float, ...] = (1.0, 0.3, 0.3)

    def with_(self, **changes) -> "PipelineConfig":
        """Return a copy with ``changes`` applied."""
        return replace(self, **changes)

    @property
    def encoder_name(self) -> str:
        if self.encoder not in ENCODERS:
            raise ValueError(f"Unknown encoder {self.encoder!r}; pick one of {sorted(ENCODERS)}")
        return ENCODERS[self.encoder]

    def embedding_fingerprint(self) -> str:
        """Hash of every setting that changes the embedding matrix."""
        payload = {
            "encoder": self.encoder_name,
            "max_chunk_chars": self.max_chunk_chars,
            "system_max_chunks": self.system_max_chunks,
            "user_max_chunks": self.user_max_chunks,
            "segments": list(SEGMENTS),
            "use_fp16": self.use_fp16,
            "format_version": 3,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def to_json(self) -> dict:
        out = {}
        for key, value in asdict(self).items():
            out[key] = str(value) if isinstance(value, Path) else value
        return out


def fast_config(**overrides) -> PipelineConfig:
    """Static-lookup encoder: whole pipeline in seconds, no transformer forward pass."""
    return PipelineConfig(encoder="static").with_(**overrides)


def accurate_config(**overrides) -> PipelineConfig:
    """MiniLM encoder: contextual embeddings, minutes on first run, cached after."""
    return PipelineConfig(
        encoder="minilm",
        system_max_chunks=8,
        user_max_chunks=32,
    ).with_(**overrides)
