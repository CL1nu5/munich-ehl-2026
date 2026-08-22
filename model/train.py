from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset


EXPORT_DIR = Path(__file__).parents[1] / "export"
TRAIN_INPUTS_PATH = EXPORT_DIR / "train_inputs.jsonl"
TRAIN_TARGETS_PATH = EXPORT_DIR / "train_targets.jsonl"
VALIDATION_INPUTS_PATH = EXPORT_DIR / "validation_inputs.jsonl"
VALIDATION_TARGETS_PATH = EXPORT_DIR / "validation_targets.jsonl"
TARGET_FIELD = "complexity_score"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENS_PER_CHUNK = 240
CACHE_FORMAT_VERSION = 2
TRAIN_EMBEDDING_CACHE_PATH = Path(__file__).with_name("train_embeddings.pt")
VALIDATION_EMBEDDING_CACHE_PATH = Path(__file__).with_name("validation_embeddings.pt")


# load JSONL file and index records by their shared request ID
def load_jsonl_by_request_id(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            request_id = record["request_id"]
            if request_id in records:
                raise ValueError(f"Duplicate request_id {request_id!r} in {path}:{line_number}")
            records[request_id] = record
    return records


# join inputs to targets by request_id and return records, scores, and IDs
def load_training_data(
    inputs_path: Path, targets_path: Path, target_field: str
) -> tuple[list[dict], torch.Tensor, list[str]]:
    
    inputs_by_id = load_jsonl_by_request_id(inputs_path)
    targets_by_id = load_jsonl_by_request_id(targets_path)

    input_ids = set(inputs_by_id)
    target_ids = set(targets_by_id)
    if input_ids != target_ids:
        missing_targets = input_ids - target_ids
        missing_inputs = target_ids - input_ids
        raise ValueError(
            "Input/target request IDs do not match: "
            f"{len(missing_targets)} missing targets, {len(missing_inputs)} missing inputs."
        )

    request_ids = list(inputs_by_id)
    rows = [inputs_by_id[request_id] for request_id in request_ids]
    scores = torch.tensor(
        [targets_by_id[request_id][target_field] for request_id in request_ids],
        dtype=torch.float32,
    )
    return rows, scores, request_ids


def row_to_text(row: dict) -> str:
    """Turn one provided input record into the text seen by the model.

    This excludes ``request_id`` and all target-file fields, preventing label
    or identifier leakage into the prediction.
    """
    history = json.dumps(row.get("user_messages", []), ensure_ascii=False)
    return (
        f"SYSTEM:\n{row.get('system_prompt', '')}\n\n"
        f"USER:\n{row.get('user_prompt', '')}\n\n"
        f"USER MESSAGE HISTORY:\n{history}"
    )


def chunk_text(
    text: str, tokenizer: object, tokens_per_chunk: int = TOKENS_PER_CHUNK
) -> list[str]:
    """Split one long row into MiniLM-sized chunks without dropping its tail."""
    # Do not ask the tokenizer to run the model's 256-token truncation here:
    # we want the complete row, then split it ourselves.
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )
    if not token_ids:
        return [""]
    return [
        tokenizer.decode(token_ids[start : start + tokens_per_chunk])
        for start in range(0, len(token_ids), tokens_per_chunk)
    ]

# load separate training and validation data
train_rows, train_scores, train_request_ids = load_training_data(
    TRAIN_INPUTS_PATH, TRAIN_TARGETS_PATH, TARGET_FIELD
)
validation_rows, validation_scores, validation_request_ids = load_training_data(
    VALIDATION_INPUTS_PATH, VALIDATION_TARGETS_PATH, TARGET_FIELD
)
print(
    f"Loaded {len(train_rows)} training rows and {len(validation_rows)} validation rows "
    f"with target field {TARGET_FIELD}"
)

def input_file_hash(path: Path) -> str:
    """Return a content hash so caches are invalidated when inputs change."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_metadata(input_path: Path) -> dict[str, object]:
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "input_sha256": input_file_hash(input_path),
    }


# load encoder model
def load_encoder() -> SentenceTransformer:
    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    # Leave room for special tokens when each 240-token chunk is encoded.
    encoder.max_seq_length = TOKENS_PER_CHUNK
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder

# encode every row once, returns 384-dimensional vector
def embed_rows(rows: list[dict], encoder: SentenceTransformer) -> torch.Tensor:
    embeddings = []
    for index, row in enumerate(rows, start=1):
        chunks = chunk_text(row_to_text(row), encoder.tokenizer)
        chunk_embeddings = encoder.encode(
            chunks, batch_size=32, convert_to_tensor=True, show_progress_bar=False
        )
        embeddings.append(chunk_embeddings.mean(dim=0).cpu())
        if index % 50 == 0 or index == len(rows):
            print(f"Embedded {index}/{len(rows)} rows")
    return torch.stack(embeddings)


def load_or_create_embeddings(
    rows: list[dict],
    request_ids: list[str],
    input_path: Path,
    cache_path: Path,
    encoder: SentenceTransformer,
) -> torch.Tensor:
    """Load valid cached embeddings, or compute and cache them once."""
    metadata = cache_metadata(input_path)
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("metadata") == metadata and cache.get("request_ids") == request_ids:
            print(f"Loaded cached embeddings from {cache_path.name}")
            return cache["embeddings"]
        print("Embedding cache is stale; rebuilding it.")

    embeddings = embed_rows(rows, encoder)
    torch.save(
        {"metadata": metadata, "request_ids": request_ids, "embeddings": embeddings},
        cache_path,
    )
    print(f"Saved embeddings to {cache_path.name}")
    return embeddings


encoder = load_encoder()
train_x = load_or_create_embeddings(
    train_rows,
    train_request_ids,
    TRAIN_INPUTS_PATH,
    TRAIN_EMBEDDING_CACHE_PATH,
    encoder,
)
validation_x = load_or_create_embeddings(
    validation_rows,
    validation_request_ids,
    VALIDATION_INPUTS_PATH,
    VALIDATION_EMBEDDING_CACHE_PATH,
    encoder,
)
train_y = train_scores
validation_y = validation_scores

# Standardising the target makes the optimiser's learning rate stable.
target_mean = train_y.mean()
target_std = train_y.std().clamp_min(1e-6)
train_y_normalized = (train_y - target_mean) / target_std

# Small nonlinear regression head. MiniLM remains frozen.
head = nn.Sequential(
    nn.Linear(384, 64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
)
loss_fn = nn.HuberLoss()
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
train_loader = DataLoader(TensorDataset(train_x, train_y_normalized), batch_size=32, shuffle=True)

for epoch in range(1, 201):
    head.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = head(batch_x).squeeze(1)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        head.eval()
        with torch.no_grad():
            validation_predictions = head(validation_x).squeeze(1) * target_std + target_mean
            validation_mae = (validation_predictions - validation_y).abs().mean()
            prediction_centered = validation_predictions - validation_predictions.mean()
            target_centered = validation_y - validation_y.mean()
            correlation = (
                (prediction_centered * target_centered).sum()
                / (prediction_centered.square().sum().sqrt() * target_centered.square().sum().sqrt()).clamp_min(1e-12)
            )
        print(
            f"Epoch {epoch:3d} | validation MAE: {validation_mae.item():.3f} "
            f"| Pearson r: {correlation.item():.3f}"
        )
