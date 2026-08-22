"""Fast text embedding for routing-time request text.

The corpus is ~49M characters of prompt text, but most of it is repeated: the
Viktor system prompt is near-identical across requests, and ``user_messages`` is
byte-for-byte the ``"\\n\\n"``-join that already forms ``user_prompt``. Four
choices do the work here:

1. **Only non-redundant text is embedded.** ``user_messages`` is dropped.
2. **Chunks are packed on paragraph boundaries.** Two prompts stay in phase for
   as long as they agree, so a shared prefix — which is most of what repeats
   here — collapses to shared chunks, as does any block too long to pack. This
   is not a general alignment guarantee: once two prompts differ in length
   before a shared run, the run is chunked differently. Content-defined
   boundaries were measured on the export and recovered nothing over this
   (1.58x either way), so the simpler rule stands.
3. **Identical chunks are embedded once** and the result is scattered back to
   every row that referenced them.
4. **Per-segment chunk budgets** bound the cost of a pathologically long prompt
   without truncating it: chunks are sampled across the whole segment.

The result is cached on disk keyed by a fingerprint of the settings *and* the
chunk text, so a re-run costs nothing and a settings change invalidates cleanly.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SEGMENTS, PipelineConfig


def segment_texts(row: dict) -> dict[str, str]:
    """Split one input row into the text segments that get embedded.

    Only predictor-safe fields are read: the system prompt and the user text.
    ``request_id``, targets and audit metadata never reach the encoder.
    ``user_messages`` is deliberately ignored — it is exactly the join that
    produced ``user_prompt``, so embedding it would double the cost for no signal.
    """
    return {
        "system": row.get("system_prompt", "") or "",
        "user": row.get("user_prompt", "") or "",
    }


def pack_chunks(text: str, max_chars: int) -> list[str]:
    """Greedily pack paragraph blocks into chunks of at most ``max_chars``.

    Packing on ``\\n\\n`` boundaries (rather than slicing at fixed offsets) is what
    makes deduplication work: a shared block of boilerplate lands in an identical
    chunk in every prompt that contains it. Blocks longer than ``max_chars`` are
    split at fixed offsets as a fallback.
    """
    if not text:
        return []
    chunks: list[str] = []
    buffer = ""
    for paragraph in text.split("\n\n"):
        if len(paragraph) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(
                paragraph[start : start + max_chars]
                for start in range(0, len(paragraph), max_chars)
            )
            continue
        if buffer and len(buffer) + 2 + len(paragraph) > max_chars:
            chunks.append(buffer)
            buffer = paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
    if buffer:
        chunks.append(buffer)
    return chunks


def subsample_chunks(chunks: list[str], budget: int) -> list[str]:
    """Reduce ``chunks`` to at most ``budget`` entries, spread across the segment.

    The first and last chunk are always kept (a prompt's opening frames the task
    and its tail usually holds the live request); the remainder is sampled at an
    even stride so the middle is represented rather than truncated away.

    A ``budget`` of 0 or less means **no cap** — every chunk is kept. It never
    means "drop this segment".
    """
    if budget <= 0 or len(chunks) <= budget:
        return chunks
    if budget == 1:
        return [chunks[0]]
    positions = np.linspace(0, len(chunks) - 1, budget)
    keep = sorted({int(round(p)) for p in positions})
    return [chunks[i] for i in keep]


@dataclass
class ChunkPlan:
    """Which unique chunk each (row, segment) is the mean of."""

    unique_chunks: list[str]
    #: Flat array of indices into ``unique_chunks``, grouped row-major by segment.
    flat_indices: np.ndarray
    #: Start offset of each group in ``flat_indices``; groups are ``n_rows * n_segments``.
    group_offsets: np.ndarray
    group_sizes: np.ndarray
    n_rows: int
    segments: tuple[str, ...]
    total_chunks: int

    @property
    def dedup_ratio(self) -> float:
        return self.total_chunks / max(1, len(self.unique_chunks))

    def stats(self) -> dict:
        return {
            "rows": self.n_rows,
            "chunks_referenced": int(self.total_chunks),
            "chunks_encoded": len(self.unique_chunks),
            "dedup_ratio": round(self.dedup_ratio, 2),
            "chars_encoded": int(sum(len(c) for c in self.unique_chunks)),
        }


def build_chunk_plan(rows: list[dict], config: PipelineConfig) -> ChunkPlan:
    """Chunk every row, deduplicate globally, and record the pooling groups."""
    budgets = {"system": config.system_max_chunks, "user": config.user_max_chunks}
    unique: dict[str, int] = {}
    flat: list[int] = []
    sizes: list[int] = []

    for row in rows:
        texts = segment_texts(row)
        for segment in SEGMENTS:
            chunks = pack_chunks(texts.get(segment, ""), config.max_chunk_chars)
            chunks = subsample_chunks(chunks, budgets.get(segment, 0))
            for chunk in chunks:
                index = unique.get(chunk)
                if index is None:
                    index = len(unique)
                    unique[chunk] = index
                flat.append(index)
            sizes.append(len(chunks))

    sizes_array = np.asarray(sizes, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes_array)[:-1]]) if len(sizes_array) else np.zeros(0, np.int64)
    return ChunkPlan(
        unique_chunks=list(unique),
        flat_indices=np.asarray(flat, dtype=np.int64),
        group_offsets=offsets.astype(np.int64),
        group_sizes=sizes_array,
        n_rows=len(rows),
        segments=SEGMENTS,
        total_chunks=int(sizes_array.sum()),
    )


def resolve_device(config: PipelineConfig) -> str:
    """Pick the device. The static encoder is a lookup — accelerator transfer costs
    more than it saves, so it stays on CPU."""
    if config.device:
        return config.device
    if config.encoder == "static":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(config: PipelineConfig):
    from sentence_transformers import SentenceTransformer

    device = resolve_device(config)
    encoder = SentenceTransformer(config.encoder_name, device=device)
    if config.encoder == "minilm":
        # Chunks are sized to this window; leave headroom for the special tokens.
        encoder.max_seq_length = max(16, config.max_chunk_chars // 4 + 16)
        if config.use_fp16 and device in {"cuda", "mps"}:
            encoder = encoder.half()
    encoder.eval()
    return encoder


def encode_unique(
    chunks: list[str],
    config: PipelineConfig,
    *,
    encoder=None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Encode the deduplicated chunk list into a float32 matrix.

    Returns the matrix and a timing dict that keeps the one-off encoder load
    separate from the per-chunk throughput, so the two are never conflated.
    """
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32), {
            "encoder_load_seconds": 0.0, "cold_encode_seconds": 0.0, "chunks_per_second": 0,
        }
    load_started = time.perf_counter()
    encoder = encoder if encoder is not None else load_encoder(config)
    load_seconds = time.perf_counter() - load_started
    device = resolve_device(config)
    if verbose:
        print(
            f"Encoding {len(chunks):,} unique chunks with {config.encoder} on {device}"
            f"{' (fp16)' if config.use_fp16 and config.encoder == 'minilm' and device != 'cpu' else ''}"
        )
    started = time.perf_counter()
    matrix = encoder.encode(
        chunks,
        batch_size=config.batch_size,
        convert_to_numpy=True,
        show_progress_bar=verbose,
        normalize_embeddings=False,
    )
    elapsed = time.perf_counter() - started
    if verbose:
        print(f"Loaded encoder in {load_seconds:.1f}s, encoded in {elapsed:.1f}s "
              f"({len(chunks) / max(elapsed, 1e-9):,.0f} chunks/s)")
    return np.asarray(matrix, dtype=np.float32), {
        "encoder_load_seconds": round(load_seconds, 2),
        "cold_encode_seconds": round(elapsed, 2),
        "chunks_per_second": round(len(chunks) / max(elapsed, 1e-9)),
    }


def pool_plan(plan: ChunkPlan, matrix: np.ndarray) -> np.ndarray:
    """Mean-pool each group, L2-normalise per segment, and concatenate.

    Groups are contiguous in ``flat_indices`` by construction, so the whole
    reduction is a single ``reduceat`` rather than a Python loop over rows.
    """
    dim = matrix.shape[1]
    n_groups = len(plan.group_sizes)
    pooled = np.zeros((n_groups, dim), dtype=np.float32)

    nonempty = plan.group_sizes > 0
    if nonempty.any():
        gathered = matrix[plan.flat_indices]
        sums = np.add.reduceat(gathered, plan.group_offsets[nonempty], axis=0)
        pooled[nonempty] = sums / plan.group_sizes[nonempty][:, None]

    # (rows, segments, dim) -> per-segment L2 norm -> (rows, segments * dim)
    per_segment = pooled.reshape(plan.n_rows, len(plan.segments), dim)
    norms = np.linalg.norm(per_segment, axis=2, keepdims=True)
    per_segment = per_segment / np.maximum(norms, 1e-9)
    return per_segment.reshape(plan.n_rows, len(plan.segments) * dim)


def _cache_path(config: PipelineConfig, plan: ChunkPlan, tag: str) -> Path:
    digest = hashlib.sha256()
    digest.update(config.embedding_fingerprint().encode())
    for chunk in plan.unique_chunks:
        digest.update(chunk.encode("utf-8", "ignore"))
        digest.update(b"\x00")
    digest.update(plan.flat_indices.tobytes())
    digest.update(plan.group_sizes.tobytes())
    return config.cache_dir / f"{tag}-{config.encoder}-{digest.hexdigest()[:20]}.npy"


def embed_rows(
    rows: list[dict],
    config: PipelineConfig,
    *,
    tag: str = "rows",
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Embed ``rows`` into a ``(len(rows), segments * dim)`` matrix.

    Returns the matrix and a stats dict (chunk counts, cache hit, seconds).
    """
    started = time.perf_counter()
    plan = build_chunk_plan(rows, config)
    stats = plan.stats()

    cache_path = _cache_path(config, plan, tag)
    sidecar = cache_path.with_suffix(".json")
    if config.use_cache and cache_path.exists():
        matrix = np.load(cache_path)
        stats.update(cached=True, seconds=round(time.perf_counter() - started, 2),
                     dim=int(matrix.shape[1]))
        # The first, uncached run is the honest cost of this encoder; report it even
        # on a cache hit so a benchmark table cannot make a slow encoder look free.
        if sidecar.exists():
            stats.update(json.loads(sidecar.read_text()))
        if verbose:
            print(f"Loaded cached embeddings {cache_path.name} {tuple(matrix.shape)}")
        return matrix, stats

    unique_matrix, cold = encode_unique(plan.unique_chunks, config, verbose=verbose)
    matrix = pool_plan(plan, unique_matrix)
    if config.use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
        sidecar.write_text(json.dumps(cold))
    stats.update(cached=False, seconds=round(time.perf_counter() - started, 2),
                 dim=int(matrix.shape[1]), **cold)
    return matrix, stats
