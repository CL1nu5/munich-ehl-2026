#!/usr/bin/env python3
"""Build a leakage-safe prompt-complexity dataset from Viktor request logs.

The raw export contains one JSON object per observed LLM request and has no
``output`` or ``usage`` field. Later requests can contain earlier requests as an
exact input prefix. This script therefore:

1. reconstructs prefix-linked request chains;
2. computes transparent prompt and observed-execution complexity components;
3. assigns all rows with the same opening prompt to exactly one split; and
4. writes separate ML inputs, targets, and audit metadata.

Only system/developer and user text is written to the ML input files. Logged
model, source location, future trajectory information, and observed execution
metrics are target/audit data and cannot leak into model inputs.

Usage:
    python scripts/build_complexity_dataset.py export/
    python scripts/build_complexity_dataset.py export/ \
        --output-dir results/complexity_dataset --seed 42

The generated files contain proprietary challenge data. Keep them under the
gitignored ``results/`` directory and do not redistribute them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


INTRINSIC_WEIGHTS = {
    "reasoning_depth": 0.30,
    "tool_action_complexity": 0.25,
    "requirements_constraints": 0.20,
    "domain_difficulty": 0.15,
    "context_modality": 0.10,
}

OBSERVED_WEIGHTS = {
    "execution_depth": 0.40,
    "friction_recovery": 0.35,
    "produced_work": 0.15,
    "tool_breadth": 0.10,
}

REASONING_TERMS = (
    "analyze", "analyse", "reason", "compare", "evaluate", "assess",
    "optimize", "optimise", "design", "architecture", "strategy",
    "derive", "prove", "debug", "investigate", "trade-off", "tradeoff",
    "plan", "decide", "explain why", "root cause", "step by step",
)
ACTION_TERMS = (
    "create", "build", "implement", "update", "fix", "write", "generate",
    "produce", "download", "upload", "extract", "send", "query", "deploy",
    "run", "execute", "install", "configure", "integrate", "migrate",
    "schedule", "search", "fetch", "commit", "push", "refactor",
)
CONSTRAINT_TERMS = (
    "must", "should", "do not", "don't", "without", "only", "exactly",
    "at least", "at most", "no more than", "ensure", "required", "format",
    "include", "exclude", "keep", "preserve", "avoid", "never", "before",
    "after", "while", "unless", "if", "otherwise", "edge case",
)
DOMAIN_TERMS = (
    "api", "database", "sql", "schema", "algorithm", "distributed",
    "concurrency", "authentication", "authorization", "security",
    "cryptography", "legal", "contract", "regulation", "compliance",
    "financial", "accounting", "medical", "clinical", "statistical",
    "causal", "machine learning", "neural", "compiler", "infrastructure",
    "kubernetes", "docker", "terraform", "cloud", "architecture",
    "optimization", "mathematical", "proof", "scientific", "engineering",
)
CORRECTION_TERMS = (
    "try again", "retry", "fix", "incorrect", "wrong", "not what i",
    "instead", "you missed", "doesn't work", "does not work", "failed",
    "please correct", "correction",
)
DIRECTIVE_TERMS = tuple(dict.fromkeys(REASONING_TERMS + ACTION_TERMS + CONSTRAINT_TERMS))

STEP_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
FILE_RE = re.compile(r"(?:^|\s)(?:[\w.-]+/)+[\w.-]+")
ERROR_RE = re.compile(
    r"(?:\btraceback\b|\bexception\b|\bcommand failed\b|"
    r"\bexit(?:ed)? (?:code|status)\s*[1-9]\d*\b|"
    r"[\"']error[\"']\s*:|\bstatus\s*[:=]\s*[45]\d\d\b)",
    re.IGNORECASE,
)
DIRECTIVE_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(phrase).replace(r"\ ", r"\s+")
        for phrase in sorted(DIRECTIVE_TERMS, key=len, reverse=True)
    )
    + r")(?!\w)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*>", re.IGNORECASE)
UUID_RE = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.IGNORECASE)
LONG_HEX_RE = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
SERIALIZED_LINE_RE = re.compile(r"^\s*(?:[{}\[\],]|[\"'][^\"']+[\"']\s*:|[\w.-]+\s*:)" )


@dataclass
class RequestRecord:
    source: str
    line_no: int
    request: dict[str, Any]
    request_id: str
    split_group_id: str
    trajectory_id: str = ""
    request_position: int = 0
    trajectory_size: int = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def short_hash(value: Any, length: int = 24) -> str:
    raw = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def iter_requests(export_dir: Path) -> Iterable[RequestRecord]:
    chunks = sorted(export_dir.glob("*.jsonl"))
    if not chunks:
        raise FileNotFoundError(f"No *.jsonl files found in {export_dir}")

    for chunk in chunks:
        with chunk.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                request = json.loads(line)
                expected = {"model", "input", "tools"}
                if set(request) != expected:
                    raise ValueError(
                        f"{chunk.name}:{line_no}: expected exactly {sorted(expected)}, "
                        f"got {sorted(request)}"
                    )
                if not isinstance(request["input"], list):
                    raise TypeError(f"{chunk.name}:{line_no}: input must be a list")
                # The system prompt can contain request-specific runtime context.
                # The first user message is the stable task identity described by
                # the challenge, and we hash its complete text (never a truncated
                # prefix) to avoid the starter loader's long-prompt collisions.
                first_user = first_user_message_text(request["input"])
                group_id = short_hash(
                    {"first_user": first_user}
                    if first_user
                    else {"no_user_opening": opening_items(request["input"])}
                )
                yield RequestRecord(
                    source=chunk.name,
                    line_no=line_no,
                    request=request,
                    request_id=f"{chunk.name}:{line_no}",
                    split_group_id=group_id,
                )


def opening_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the stable opening prefix through the first user message."""
    opening: list[dict[str, Any]] = []
    for item in items:
        opening.append(item)
        if item.get("role") == "user":
            return opening
    # Malformed/no-user requests stay isolated using their complete input.
    return list(items)


def first_user_message_text(items: Sequence[dict[str, Any]]) -> str:
    for item in items:
        if item.get("role") == "user":
            return content_to_text(item.get("content"))
    return ""


def is_strict_item_prefix(shorter: Sequence[Any], longer: Sequence[Any]) -> bool:
    return len(shorter) < len(longer) and list(shorter) == list(longer[: len(shorter)])


def reconstruct_chains(records: Sequence[RequestRecord]) -> list[list[RequestRecord]]:
    """Build conservative exact-prefix chains within opening-prompt/model groups.

    Exact duplicate openings across models remain in one split via
    ``split_group_id`` but are separate observed trajectories. Ambiguous branches
    become separate chains; all branches still share a split, preventing leakage.
    """
    candidates: dict[tuple[str, str], list[RequestRecord]] = defaultdict(list)
    for record in records:
        candidates[(record.split_group_id, record.request["model"])].append(record)

    all_chains: list[list[RequestRecord]] = []
    for (group_id, model), group in sorted(candidates.items()):
        chains: list[list[RequestRecord]] = []
        ordered = sorted(group, key=lambda r: (len(r.request["input"]), r.request_id))
        for record in ordered:
            compatible = [
                chain for chain in chains
                if is_strict_item_prefix(chain[-1].request["input"], record.request["input"])
            ]
            if compatible:
                chain = max(compatible, key=lambda c: len(c[-1].request["input"]))
                chain.append(record)
            else:
                chains.append([record])

        for chain_index, chain in enumerate(chains):
            trajectory_id = short_hash(f"{group_id}:{model}:{chain_index}")
            for position, record in enumerate(chain):
                record.trajectory_id = trajectory_id
                record.request_position = position
                record.trajectory_size = len(chain)
            all_chains.append(chain)
    return all_chains


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif part.get("type") in {"input_image", "image", "image_url"}:
            parts.append("[IMAGE]")
    return "\n".join(parts)


def prompt_texts(request: dict[str, Any]) -> tuple[str, list[str]]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for item in request["input"]:
        text = content_to_text(item.get("content"))
        if not text:
            continue
        if item.get("role") in {"system", "developer"}:
            system_parts.append(text)
        elif item.get("role") == "user":
            user_parts.append(text)
    return "\n\n".join(system_parts), user_parts


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalized_prompt_tokens(text: str) -> list[str]:
    normalized = text.casefold()
    normalized = PLACEHOLDER_RE.sub(" placeholder ", normalized)
    normalized = URL_RE.sub(" url ", normalized)
    normalized = UUID_RE.sub(" identifier ", normalized)
    normalized = LONG_HEX_RE.sub(" identifier ", normalized)
    normalized = NUMBER_RE.sub(" number ", normalized)
    return TOKEN_RE.findall(normalized)


def prompt_shingles(text: str, width: int = 5) -> set[str]:
    tokens = normalized_prompt_tokens(text)
    if len(tokens) < width:
        return set(tokens) or {"<empty>"}
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def minhash_signature(shingles: set[str], permutations: int = 64) -> tuple[int, ...]:
    prime = 18_446_744_073_709_551_557
    bases = [stable_u64(shingle) % prime for shingle in shingles]
    # Deterministic bottom-hash sampling bounds work on very large context dumps
    # without changing the confirmation step, which still uses full shingle sets.
    if len(bases) > 512:
        bases = sorted(bases)[:512]
    signature: list[int] = []
    for index in range(permutations):
        digest = hashlib.sha256(f"minhash:{index}".encode("utf-8")).digest()
        multiplier = int.from_bytes(digest[:8], "big") % (prime - 1) + 1
        offset = int.from_bytes(digest[8:16], "big") % prime
        signature.append(min((multiplier * base + offset) % prime for base in bases))
    return tuple(signature)


def assign_semantic_split_groups(records: Sequence[RequestRecord]) -> None:
    """Cluster exact and near-duplicate task openings for split isolation."""
    texts = [first_user_message_text(record.request["input"]) for record in records]
    shingles = [prompt_shingles(text) for text in texts]
    signatures = [minhash_signature(parts) for parts in shingles]
    union_find = UnionFind(len(records))

    exact: dict[str, int] = {}
    for index, text in enumerate(texts):
        normalized = " ".join(normalized_prompt_tokens(text))
        if normalized in exact:
            union_find.union(index, exact[normalized])
        else:
            exact[normalized] = index

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        for band in range(8):
            start = band * 8
            buckets[(band, signature[start : start + 8])].append(index)

    checked: set[tuple[int, int]] = set()
    for candidates in buckets.values():
        for offset, left in enumerate(candidates):
            for right in candidates[offset + 1 :]:
                pair = (min(left, right), max(left, right))
                if pair in checked:
                    continue
                checked.add(pair)
                intersection = len(shingles[left] & shingles[right])
                union = len(shingles[left] | shingles[right])
                smaller = min(len(shingles[left]), len(shingles[right]))
                jaccard = intersection / union if union else 1.0
                containment = intersection / smaller if smaller else 1.0
                if jaccard >= 0.80 or containment >= 0.90:
                    union_find.union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members[union_find.find(index)].append(index)
    for indices in members.values():
        cluster_id = short_hash({
            "semantic_cluster": sorted(records[index].split_group_id for index in indices)
        })
        for index in indices:
            records[index].split_group_id = cluster_id


@lru_cache(maxsize=None)
def phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.casefold()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def phrase_count(text: str, phrases: Sequence[str]) -> int:
    return sum(len(phrase_pattern(phrase).findall(text)) for phrase in phrases)


def unique_phrase_count(text: str, phrases: Sequence[str]) -> int:
    return sum(bool(phrase_pattern(phrase).search(text)) for phrase in phrases)


def estimated_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def normalized_paragraph(paragraph: str) -> str:
    tokens = normalized_prompt_tokens(paragraph)
    return " ".join(tokens)


def common_system_paragraphs(
    prompt_text_by_id: dict[str, tuple[str, list[str]]], train_ids: set[str]
) -> set[str]:
    counts: Counter[str] = Counter()
    for request_id in train_ids:
        system_prompt = prompt_text_by_id[request_id][0]
        paragraphs = {
            normalized_paragraph(part)
            for part in re.split(r"\n\s*\n", system_prompt)
            if len(normalized_paragraph(part)) >= 40
        }
        counts.update(paragraphs)
    threshold = max(2, math.ceil(0.20 * max(1, len(train_ids))))
    return {paragraph for paragraph, count in counts.items() if count >= threshold}


def filtered_system_text(system_prompt: str, boilerplate: set[str]) -> str:
    kept = [
        part for part in re.split(r"\n\s*\n", system_prompt)
        if normalized_paragraph(part) not in boilerplate
    ]
    return "\n\n".join(kept)


def segment_weight(segment: str) -> float:
    stripped = segment.strip()
    if not stripped:
        return 0.0
    lines = stripped.splitlines()
    quoted = sum(line.lstrip().startswith(">") for line in lines)
    serialized = sum(bool(SERIALIZED_LINE_RE.search(line)) for line in lines)
    reference_like = "```" in stripped or quoted > len(lines) / 2 or serialized > len(lines) / 2
    if reference_like:
        return 0.10
    directive = stripped.count("?") + int(bool(DIRECTIVE_RE.search(stripped)))
    if len(stripped) > 2_000 and directive == 0:
        return 0.25
    return 1.0 if directive else 0.50


def weighted_segments(user_messages: Sequence[str]) -> list[tuple[str, float]]:
    segments: list[tuple[str, float]] = []
    for message in user_messages:
        parts = re.split(
            r"\n\s*\n|(?=^\s*(?:[-*+]\s+|\d+[.)]\s+))",
            message,
            flags=re.MULTILINE,
        )
        for part in parts:
            weight = segment_weight(part)
            if weight:
                segments.append((part, weight))
    return segments


def weighted_phrase_count(segments: Sequence[tuple[str, float]], phrases: Sequence[str]) -> float:
    return sum(weight * phrase_count(segment, phrases) for segment, weight in segments)


def weighted_unique_phrase_count(
    segments: Sequence[tuple[str, float]], phrases: Sequence[str]
) -> float:
    maxima = []
    for phrase in phrases:
        maxima.append(max((weight for segment, weight in segments if phrase_pattern(phrase).search(segment)), default=0.0))
    return sum(maxima)


def weighted_term_statistics(
    segments: Sequence[tuple[str, float]], categories: dict[str, Sequence[str]]
) -> tuple[dict[str, float], dict[str, float]]:
    tokenized = {
        category: [(phrase, tuple(TOKEN_RE.findall(phrase.casefold()))) for phrase in phrases]
        for category, phrases in categories.items()
    }
    weighted_counts = {category: 0.0 for category in categories}
    unique_weights = {
        category: {phrase: 0.0 for phrase in phrases}
        for category, phrases in categories.items()
    }
    lengths = {
        len(tokens)
        for terms in tokenized.values()
        for _, tokens in terms
        if tokens
    }
    for segment, weight in segments:
        tokens = TOKEN_RE.findall(segment.casefold())
        ngrams = {
            length: Counter(
                tuple(tokens[index : index + length])
                for index in range(max(0, len(tokens) - length + 1))
            )
            for length in lengths
        }
        for category, terms in tokenized.items():
            for phrase, phrase_tokens in terms:
                occurrences = ngrams[len(phrase_tokens)][phrase_tokens] if phrase_tokens else 0
                if occurrences:
                    weighted_counts[category] += weight * occurrences
                    unique_weights[category][phrase] = max(
                        unique_weights[category][phrase], weight
                    )
    unique_counts = {
        category: sum(values.values()) for category, values in unique_weights.items()
    }
    return weighted_counts, unique_counts


def prompt_raw_metrics(
    system_prompt: str,
    user_messages: Sequence[str],
    system_boilerplate: set[str] | None = None,
) -> dict[str, float]:
    user_prompt = "\n\n".join(user_messages)
    segments = weighted_segments(user_messages)
    instruction_tokens = max(50.0, sum(weight * estimated_tokens(segment) for segment, weight in segments))
    term_counts, unique_counts = weighted_term_statistics(segments, {
        "reasoning": REASONING_TERMS,
        "actions": ACTION_TERMS,
        "constraints": CONSTRAINT_TERMS,
        "domains": DOMAIN_TERMS,
        "conditionals": ("if", "unless", "otherwise", "depending"),
    })
    reasoning_terms = term_counts["reasoning"]
    actions = term_counts["actions"]
    constraints = term_counts["constraints"]
    domain_terms = term_counts["domains"]
    unique_reasoning = unique_counts["reasoning"]
    unique_actions = unique_counts["actions"]
    unique_constraints = unique_counts["constraints"]
    unique_domains = unique_counts["domains"]
    steps = len(STEP_RE.findall(user_prompt))
    questions = user_prompt.count("?")
    conditionals = term_counts["conditionals"]
    code_blocks = user_prompt.count("```") // 2
    urls = len(URL_RE.findall(user_prompt))
    file_refs = len(FILE_RE.findall(user_prompt))
    image_markers = user_prompt.count("[IMAGE]") + user_prompt.count("[base64 image redacted]")
    identifier_count = len(re.findall(r"\b\w+(?:_\w+)+\b|\b[A-Za-z]+\.[A-Za-z0-9]+\b", user_prompt))
    reasoning_density = 1_000 * reasoning_terms / instruction_tokens
    action_density = 1_000 * actions / instruction_tokens
    constraint_density = 1_000 * constraints / instruction_tokens
    domain_density = 1_000 * domain_terms / instruction_tokens
    task_system = filtered_system_text(system_prompt, system_boilerplate or set())

    return {
        "reasoning_depth": float(
            unique_reasoning + 0.15 * reasoning_density + 0.5 * math.log1p(steps)
            + 0.25 * math.log1p(questions) + 0.25 * conditionals
        ),
        "tool_action_complexity": float(
            unique_actions + 0.12 * action_density + 0.5 * math.log1p(code_blocks + urls + file_refs)
            + image_markers
        ),
        "requirements_constraints": float(
            unique_constraints + 0.15 * constraint_density + 0.30 * unique_actions
            + 0.40 * math.log1p(steps)
        ),
        "domain_difficulty": float(
            unique_domains + 0.15 * domain_density + 0.20 * math.log1p(identifier_count)
            + 0.50 * math.log1p(code_blocks)
        ),
        "context_modality": float(
            estimated_tokens(user_prompt)
            + 0.05 * estimated_tokens(task_system)
            + 250 * image_markers
            + 30 * (urls + code_blocks)
        ),
        "system_est_tokens": float(estimated_tokens(system_prompt)),
        "user_est_tokens": float(estimated_tokens(user_prompt)),
        "user_message_count": float(len(user_messages)),
        "step_marker_count": float(steps),
        "question_count": float(questions),
        "action_term_count": float(actions),
        "constraint_term_count": float(constraints),
        "domain_term_count": float(domain_terms),
        "image_count": float(image_markers),
        "url_count": float(urls),
        "code_block_count": float(code_blocks),
    }


def item_payload_text(item: dict[str, Any]) -> str:
    if item.get("role") is not None:
        return content_to_text(item.get("content"))
    for key in ("output", "arguments", "input"):
        value = item.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return canonical_json(value)
    return ""


def tool_signature(item: dict[str, Any]) -> str:
    return short_hash({"name": item.get("name", ""), "arguments": item.get("arguments", "")})


def structured_error(value: Any) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return structured_error(json.loads(stripped))
        except (json.JSONDecodeError, TypeError):
            return bool(ERROR_RE.search(stripped))
    if isinstance(value, list):
        return any(structured_error(item) for item in value)
    if not isinstance(value, dict):
        return False
    for key, child in value.items():
        lowered = str(key).casefold()
        if lowered in {"error", "exception", "traceback"} and child not in (None, "", False, 0, []):
            return True
        if lowered in {"success", "ok"} and child is False:
            return True
        if lowered in {"exit_code", "returncode"} and isinstance(child, int) and child != 0:
            return True
        if lowered in {"status", "status_code"}:
            try:
                if int(child) >= 400:
                    return True
            except (TypeError, ValueError):
                pass
        if isinstance(child, (dict, list)) and structured_error(child):
            return True
    return False


def tool_round_metrics(items: Sequence[dict[str, Any]]) -> tuple[int, int]:
    rounds = 0
    calls_in_round = 0
    max_parallel = 0
    output_since_call = False
    for item in items:
        item_type = item.get("type")
        if item_type in {"function_call", "custom_tool_call"}:
            if rounds == 0 or output_since_call:
                rounds += 1
                calls_in_round = 0
                output_since_call = False
            calls_in_round += 1
            max_parallel = max(max_parallel, calls_in_round)
        elif item_type in {"function_call_output", "custom_tool_call_output"}:
            output_since_call = True
    return rounds, max_parallel


def observed_raw_metrics(chain: Sequence[RequestRecord]) -> dict[str, float]:
    # The longest snapshot contains the richest visible execution history.
    items = chain[-1].request["input"]
    tool_calls = [
        item for item in items
        if item.get("type") in {"function_call", "custom_tool_call"}
    ]
    tool_outputs = [
        item for item in items
        if item.get("type") in {"function_call_output", "custom_tool_call_output"}
    ]
    assistant_items = [
        item for item in items
        if item.get("role") == "assistant"
    ]
    names = [str(item.get("name") or item.get("type")) for item in tool_calls]
    output_by_call = {
        str(item.get("call_id")): item for item in tool_outputs if item.get("call_id") is not None
    }
    call_failed = {
        str(call.get("call_id")): structured_error(output_by_call[str(call.get("call_id"))].get("output"))
        for call in tool_calls
        if call.get("call_id") is not None and str(call.get("call_id")) in output_by_call
    }
    tool_errors = sum(call_failed.values())
    repeated_calls = 0
    recent: list[tuple[str, bool]] = []
    for call in tool_calls:
        signature = tool_signature(call)
        if any(previous == signature and failed for previous, failed in recent[-3:]):
            repeated_calls += 1
        recent.append((signature, call_failed.get(str(call.get("call_id")), False)))

    _, users = prompt_texts(chain[-1].request)
    correction_turns = sum(
        phrase_count(message, CORRECTION_TERMS) > 0 for message in users[1:]
    )
    produced_tokens = sum(min(1_000, estimated_tokens(item_payload_text(item))) for item in assistant_items)
    produced_tokens += sum(
        min(1_000, estimated_tokens(str(item.get("arguments", "")))) for item in tool_calls
    )

    snapshots = len(chain)
    unique_tools = len(set(names))
    rounds, max_parallel = tool_round_metrics(items)
    return {
        "execution_depth": float(
            rounds + 0.25 * max(0, max_parallel - 1) + max(0, snapshots - 1)
        ),
        "friction_recovery": float(2 * tool_errors + repeated_calls + correction_turns),
        "produced_work": float(produced_tokens),
        "tool_breadth": float(unique_tools),
        "snapshot_count": float(snapshots),
        "tool_call_count": float(len(tool_calls)),
        "tool_output_count": float(len(tool_outputs)),
        "assistant_history_count": float(len(assistant_items)),
        "unique_tool_count": float(unique_tools),
        "tool_error_count": float(tool_errors),
        "repeated_call_count": float(repeated_calls),
        "correction_turn_count": float(correction_turns),
        "recovered_work_est_tokens": float(produced_tokens),
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def scaling_caps(metric_rows: Sequence[dict[str, float]], names: Sequence[str]) -> dict[str, float]:
    caps: dict[str, float] = {}
    for name in names:
        values = [max(0.0, row[name]) for row in metric_rows]
        cap = percentile(values, 0.95)
        if cap <= 0:
            cap = max(values, default=1.0)
        caps[name] = max(cap, 1.0)
    return caps


def log_normalize(value: float, cap: float) -> float:
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(max(cap, 1.0)))


def normalized_components(
    raw: dict[str, float], caps: dict[str, float], names: Sequence[str]
) -> dict[str, float]:
    return {name: round(log_normalize(raw[name], caps[name]), 6) for name in names}


def weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    return sum(components[name] * weight for name, weight in weights.items())


def balanced_group_splits(
    records: Sequence[RequestRecord],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    sizes = Counter(record.split_group_id for record in records)
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    targets = {split: ratio * len(records) for split, ratio in ratios.items()}
    counts = {split: 0 for split in ratios}
    ordered = sorted(
        sizes,
        key=lambda group: (
            -sizes[group],
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
        ),
    )
    assignments: dict[str, str] = {}
    for group in ordered:
        split = max(
            ratios,
            key=lambda name: (
                (targets[name] - counts[name]) / max(targets[name], 1.0),
                -list(ratios).index(name),
            ),
        )
        assignments[group] = split
        counts[split] += sizes[group]
    return assignments


def perturbed_weight_scenarios(
    seed: int, count: int = 32
) -> list[tuple[dict[str, float], dict[str, float], float]]:
    generator = random.Random(seed + 91_337)
    scenarios = []
    for _ in range(count):
        intrinsic = {
            name: weight * generator.uniform(0.8, 1.2)
            for name, weight in INTRINSIC_WEIGHTS.items()
        }
        observed = {
            name: weight * generator.uniform(0.8, 1.2)
            for name, weight in OBSERVED_WEIGHTS.items()
        }
        intrinsic_total = sum(intrinsic.values())
        observed_total = sum(observed.values())
        intrinsic = {name: value / intrinsic_total for name, value in intrinsic.items()}
        observed = {name: value / observed_total for name, value in observed.items()}
        scenarios.append((intrinsic, observed, generator.uniform(0.8, 1.2)))
    return scenarios


def complexity_band(score: float) -> str:
    if score <= 35:
        return "low"
    if score <= 70:
        return "medium"
    return "high"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def validate_ratios(train: float, validation: float, test: float) -> None:
    if min(train, validation, test) <= 0:
        raise ValueError("All split ratios must be greater than zero")
    if not math.isclose(train + validation + test, 1.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")


def build_dataset(
    export_dir: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Any]:
    validate_ratios(train_ratio, validation_ratio, test_ratio)
    records = list(iter_requests(export_dir))
    # Reconstruct exact chains before semantic split clustering changes the
    # split-group ids shared by near-duplicate but distinct tasks.
    chains = reconstruct_chains(records)
    assign_semantic_split_groups(records)
    split_by_group = balanced_group_splits(
        records, seed, train_ratio, validation_ratio, test_ratio
    )

    prompt_text_by_id: dict[str, tuple[str, list[str]]] = {}
    for record in records:
        prompt_text_by_id[record.request_id] = prompt_texts(record.request)

    train_ids = {
        record.request_id
        for record in records
        if split_by_group[record.split_group_id] == "train"
    }
    system_boilerplate = common_system_paragraphs(prompt_text_by_id, train_ids)
    prompt_raw_by_id: dict[str, dict[str, float]] = {}
    for record in records:
        system_prompt, users = prompt_text_by_id[record.request_id]
        prompt_raw_by_id[record.request_id] = prompt_raw_metrics(
            system_prompt, users, system_boilerplate
        )

    intrinsic_names = list(INTRINSIC_WEIGHTS)
    prompt_caps = scaling_caps(
        [prompt_raw_by_id[request_id] for request_id in train_ids], intrinsic_names
    )

    observed_raw_by_trajectory: dict[str, dict[str, float]] = {}
    for chain in chains:
        observed_raw_by_trajectory[chain[0].trajectory_id] = observed_raw_metrics(chain)
    observed_names = list(OBSERVED_WEIGHTS)
    train_trajectory_ids = {
        record.trajectory_id
        for record in records
        if split_by_group[record.split_group_id] == "train"
    }
    observed_caps = scaling_caps(
        [observed_raw_by_trajectory[trajectory] for trajectory in train_trajectory_ids],
        observed_names,
    )

    prepared: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: r.request_id):
        system_prompt, users = prompt_text_by_id[record.request_id]
        prompt_raw = prompt_raw_by_id[record.request_id]
        prompt_components = normalized_components(prompt_raw, prompt_caps, intrinsic_names)
        intrinsic = weighted_score(prompt_components, INTRINSIC_WEIGHTS)

        observed_raw = observed_raw_by_trajectory[record.trajectory_id]
        observed_components = normalized_components(observed_raw, observed_caps, observed_names)
        observed = weighted_score(observed_components, OBSERVED_WEIGHTS)
        history_evidence = (
            observed_raw["tool_call_count"] > 0
            or observed_raw["tool_output_count"] > 0
            or observed_raw["assistant_history_count"] > 0
        )
        if record.trajectory_size > 1:
            observed_weight = 0.60
        elif history_evidence:
            observed_weight = 0.35
        else:
            observed_weight = 0.0
        combined = (1 - observed_weight) * intrinsic + observed_weight * observed
        prepared.append({
            "record": record,
            "system_prompt": system_prompt,
            "users": users,
            "prompt_raw": prompt_raw,
            "prompt_components": prompt_components,
            "intrinsic": intrinsic,
            "observed_raw": observed_raw,
            "observed_components": observed_components,
            "observed": observed,
            "observed_weight": observed_weight,
            "combined": combined,
        })

    scenarios = perturbed_weight_scenarios(seed)

    combined_rows: list[dict[str, Any]] = []
    for row in prepared:
        record = row["record"]
        system_prompt = row["system_prompt"]
        users = row["users"]
        prompt_raw = row["prompt_raw"]
        prompt_components = row["prompt_components"]
        intrinsic = row["intrinsic"]
        observed_raw = row["observed_raw"]
        observed_components = row["observed_components"]
        observed = row["observed"]
        observed_weight = row["observed_weight"]
        score = round(100 * row["combined"], 3)

        perturbed_scores = []
        for intrinsic_weights, observed_weights, observed_factor in scenarios:
            variant_intrinsic = weighted_score(prompt_components, intrinsic_weights)
            variant_observed = weighted_score(observed_components, observed_weights)
            variant_weight = min(0.80, observed_weight * observed_factor)
            variant_combined = (
                (1 - variant_weight) * variant_intrinsic + variant_weight * variant_observed
            )
            perturbed_scores.append(100 * variant_combined)
        score_spread = percentile(perturbed_scores, 0.90) - percentile(perturbed_scores, 0.10)
        strong_evidence = (
            record.trajectory_size > 1
            or (
                observed_raw["tool_output_count"] >= 3
                and observed_raw["assistant_history_count"] >= 1
            )
        )
        agreement = abs(100 * intrinsic - 100 * observed)
        if strong_evidence and agreement <= 20 and score_spread <= 10:
            confidence = "high"
        elif "\n\n".join(users).strip() and score_spread <= 20:
            confidence = "medium"
        else:
            confidence = "low"

        static_features = {
            key: int(value) if float(value).is_integer() else round(value, 4)
            for key, value in prompt_raw.items()
            if key not in intrinsic_names
        }
        target = {
            "request_id": record.request_id,
            "complexity_score": score,
            "complexity_band": complexity_band(score),
            "intrinsic_complexity": round(100 * intrinsic, 3),
            "observed_difficulty": round(100 * observed, 3),
            "observed_weight": observed_weight,
            "label_confidence": confidence,
            "intrinsic_components": prompt_components,
            "observed_components": observed_components,
            "observed_metrics": {
                key: int(value) if float(value).is_integer() else round(value, 4)
                for key, value in observed_raw.items()
            },
        }
        model_input = {
            "request_id": record.request_id,
            "system_prompt": system_prompt,
            "user_prompt": "\n\n".join(users),
            "user_messages": users,
            "static_text_features": static_features,
        }
        metadata = {
            "request_id": record.request_id,
            "source": record.source,
            "source_line": record.line_no,
            "split_group_id": record.split_group_id,
            "trajectory_id": record.trajectory_id,
            "request_position": record.request_position,
            "trajectory_size": record.trajectory_size,
            "logged_model": record.request["model"],
            "observed_uses_future_snapshot": record.request_position < record.trajectory_size - 1,
        }
        combined_rows.append({
            "request_id": record.request_id,
            "split": split_by_group[record.split_group_id],
            "input": model_input,
            "target": target,
            "metadata": metadata,
        })

    # Invariant: a prompt family/trajectory may never cross dataset splits.
    group_splits: dict[str, set[str]] = defaultdict(set)
    trajectory_splits: dict[str, set[str]] = defaultdict(set)
    for row in combined_rows:
        group_splits[row["metadata"]["split_group_id"]].add(row["split"])
        trajectory_splits[row["metadata"]["trajectory_id"]].add(row["split"])
    leaked_groups = [group for group, splits in group_splits.items() if len(splits) != 1]
    leaked_trajectories = [group for group, splits in trajectory_splits.items() if len(splits) != 1]
    if leaked_groups or leaked_trajectories:
        raise AssertionError(
            f"Split leakage detected: groups={len(leaked_groups)}, "
            f"trajectories={len(leaked_trajectories)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all.jsonl", combined_rows)

    split_counts: dict[str, int] = {}
    split_groups: dict[str, int] = {}
    split_bands: dict[str, dict[str, int]] = {}
    split_confidence: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in combined_rows if row["split"] == split]
        write_jsonl(output_dir / f"{split}_inputs.jsonl", (row["input"] for row in rows))
        write_jsonl(output_dir / f"{split}_targets.jsonl", (row["target"] for row in rows))
        write_jsonl(output_dir / f"{split}_metadata.jsonl", (row["metadata"] for row in rows))
        split_counts[split] = len(rows)
        split_groups[split] = len({row["metadata"]["split_group_id"] for row in rows})
        split_bands[split] = dict(Counter(row["target"]["complexity_band"] for row in rows))
        split_confidence[split] = dict(Counter(row["target"]["label_confidence"] for row in rows))

    models_per_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        models_per_group[record.split_group_id].add(record.request["model"])

    manifest = {
        "schema_version": 1,
        "source": str(export_dir),
        "output": str(output_dir),
        "request_count": len(records),
        "trajectory_count": len(chains),
        "split_group_count": len(group_splits),
        "opening_prompt_groups_with_multiple_models": sum(
            len(models) > 1 for models in models_per_group.values()
        ),
        "split_seed": seed,
        "requested_split_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "split_request_counts": split_counts,
        "split_group_counts": split_groups,
        "complexity_bands_by_split": split_bands,
        "label_confidence_by_split": split_confidence,
        "intrinsic_weights": INTRINSIC_WEIGHTS,
        "observed_weights": OBSERVED_WEIGHTS,
        "intrinsic_p95_scaling_caps": prompt_caps,
        "observed_p95_scaling_caps": observed_caps,
        "score_definition": {
            "linked_snapshots": "100 * (0.40 * intrinsic + 0.60 * observed)",
            "singleton_with_visible_execution_history": "100 * (0.65 * intrinsic + 0.35 * observed)",
            "singleton_without_execution_history": "100 * intrinsic",
            "band_thresholds": {"low": "<=35", "medium": "35<score<=70", "high": ">70"},
        },
        "split_safety": {
            "strategy": "exact-prefix grouping plus deterministic 64-value MinHash/LSH near-duplicate clusters",
            "same_opening_prompt_always_same_split": True,
            "exact_prefix_trajectory_always_same_split": True,
            "leaked_split_groups": len(leaked_groups),
            "leaked_trajectories": len(leaked_trajectories),
        },
        "important_limitations": [
            "The raw export has no final model output, usage, timing, or quality label.",
            "Complexity is a transparent weak-supervision proxy, not ground truth model capability.",
            "Tool errors can reflect external systems rather than task complexity.",
            "Long histories can reflect the serving model as well as intrinsic task difficulty.",
            "Logged model is audit metadata and is intentionally excluded from ML inputs.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", nargs="?", default="export", type=Path)
    parser.add_argument("--output-dir", default=Path("results/complexity_dataset"), type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        export_dir=args.export_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    print(
        f"requests={manifest['request_count']} trajectories={manifest['trajectory_count']} "
        f"split_groups={manifest['split_group_count']}"
    )
    print(f"split request counts: {manifest['split_request_counts']}")
    print(f"label confidence: {manifest['label_confidence_by_split']}")
    print("split leakage: 0 prompt groups, 0 trajectories")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
