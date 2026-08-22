"""Feature extraction for routing-time inputs (stdlib only)."""
from __future__ import annotations

import json
import re
from pathlib import Path

FEATURE_NAMES = [
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
]

ACTION_TERMS = re.compile(
    r"\b(run|create|send|download|upload|write|read|grep|search|execute|fix|build|"
    r"transcribe|summarize|delete|install|deploy|schedule|reply)\b",
    re.I,
)
CONSTRAINT_TERMS = re.compile(
    r"\b(must|never|always|do not|don't|required|only|ensure|important|critical)\b",
    re.I,
)
DOMAIN_TERMS = re.compile(
    r"\b(slack|github|pdf|excel|browser|cron|aws|api|msteams|notion|wav|audio)\b",
    re.I,
)
STEP_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:[-*]|\d+\.)\s|(?:^|\n)#+\s|<\/?(?:system|thread|available_skills)",
    re.I,
)


def _message_text(item) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if p.get("type") == "input_text")


def system_user_texts(req: dict) -> tuple[str, str, int]:
    system, user_parts = "", []
    for item in req.get("input", []):
        if item.get("type") != "message":
            continue
        if item.get("role") == "system" and not system:
            system = _message_text(item)
        elif item.get("role") == "user":
            user_parts.append(_message_text(item))
    return system, "\n".join(user_parts), len(user_parts)


def extract_from_request(req: dict) -> dict[str, int]:
    system, user, user_count = system_user_texts(req)
    combined = f"{system}\n{user}"
    return {
        "system_est_tokens": len(system) // 4,
        "user_est_tokens": len(user) // 4,
        "user_message_count": user_count,
        "step_marker_count": len(STEP_MARKERS.findall(combined)),
        "question_count": combined.count("?"),
        "action_term_count": len(ACTION_TERMS.findall(combined)),
        "constraint_term_count": len(CONSTRAINT_TERMS.findall(combined)),
        "domain_term_count": len(DOMAIN_TERMS.findall(combined)),
        "image_count": combined.lower().count("input_image")
        + len(re.findall(r"\.(?:png|jpe?g|gif|webp)", combined, re.I)),
        "url_count": len(re.findall(r"https?://", combined)),
        "code_block_count": combined.count("```"),
    }


def vectorize(features: dict) -> list[float]:
    return [float(features[name]) for name in FEATURE_NAMES]


def load_feature_lookup(datasets_dir: str | Path = "datasets") -> dict[str, dict]:
    """Official static_text_features from train/validation/test inputs."""
    lookup: dict[str, dict] = {}
    root = Path(datasets_dir)
    for name in ("train_inputs.jsonl", "validation_inputs.jsonl", "test_inputs.jsonl"):
        path = root / name
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                lookup[row["request_id"]] = row["static_text_features"]
    return lookup


def load_split_ids(datasets_dir: str | Path, split: str) -> set[str]:
    path = Path(datasets_dir) / f"{split}_targets.jsonl"
    if not path.exists():
        return set()
    return {json.loads(l)["request_id"] for l in open(path) if l.strip()}


def load_train_ids(datasets_dir: str | Path = "datasets") -> set[str]:
    return load_split_ids(datasets_dir, "train")


def load_held_out_ids(datasets_dir: str | Path = "datasets") -> set[str]:
    """request_ids in validation + test targets (excluded from router training)."""
    ids: set[str] = set()
    root = Path(datasets_dir)
    for name in ("validation_targets.jsonl", "test_targets.jsonl"):
        path = root / name
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["request_id"])
    return ids


def load_validation_feature_lookup(datasets_dir: str | Path = "datasets") -> dict[str, dict]:
    """Backward-compatible alias."""
    return load_feature_lookup(datasets_dir)


def features_for_request(
    req: dict,
    request_id: str | None = None,
    lookup: dict[str, dict] | None = None,
) -> dict[str, int]:
    if request_id and lookup and request_id in lookup:
        return dict(lookup[request_id])
    return extract_from_request(req)
