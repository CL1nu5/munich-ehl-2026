#!/usr/bin/env python3
"""Load a redacted-trajectories export and reconstruct trajectories.

The export is chunked JSONL — `trajectories_v1_<index>.jsonl.tar.gz` archives that
extract to `export/trajectories_v1_<index>.jsonl` (any `export/*.jsonl` is read). Each line is one LLM
request: `model`, `input` (Responses-format item list), `tools`. There are no
trajectory ids — requests of the same task are recovered conservatively from
exact item-prefix containment within requests that share the same first user
message. This avoids merging unrelated tasks that happen to reuse a boilerplate
opening prompt.

Usage: python scripts/load_trajectories.py export/
Importable: iter_requests, group_trajectories, est_tokens, first_user_text.
"""
import json, sys, hashlib
from pathlib import Path
from collections import Counter, defaultdict

def iter_requests(export_dir):
    """Yield (chunk_name, line_no, request) for every line of every chunk."""
    chunks = sorted(Path(export_dir).glob("*.jsonl"))
    if not chunks:
        sys.exit(f"no *.jsonl chunks found in {export_dir}")
    for p in chunks:
        with open(p) as f:
            for i, line in enumerate(f):
                if line.strip():
                    yield p.name, i, json.loads(line)

def first_user_text(req):
    """Text of the first user message — stable across all requests of a task."""
    for item in req["input"]:
        if item.get("role") == "user":
            c = item.get("content")
            if isinstance(c, str): return c
            return " ".join(p.get("text", "") for p in c if p.get("type") == "input_text")
    return ""

def group_key(req):
    """Stable candidate-group key using the complete first user message."""
    text = first_user_text(req)
    if text:
        opening = {"first_user": text}
    else:
        opening = {"no_user_input": req.get("input", [])}
    raw = json.dumps(opening, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

def _is_strict_input_prefix(shorter, longer):
    a, b = shorter["input"], longer["input"]
    return len(a) < len(b) and a == b[:len(a)]

def group_trajectories(requests):
    """Reconstruct conservative exact-prefix chains in call order.

    The first user message is only a candidate key: repeated cron prompts and
    templates can be identical across unrelated tasks. A request joins a chain
    only when the chain's latest input is an exact strict prefix of its input.
    Ambiguous branches become separate trajectories instead of being merged.
    """
    candidates = defaultdict(list)
    for req in requests:
        candidates[group_key(req)].append(req)

    trajectories = {}
    for base_key, candidate_reqs in sorted(candidates.items()):
        chains = []
        ordered = sorted(
            candidate_reqs,
            key=lambda r: (
                len(r["input"]),
                hashlib.sha256(
                    json.dumps(r["input"], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            ),
        )
        for req in ordered:
            compatible = [chain for chain in chains if _is_strict_input_prefix(chain[-1], req)]
            if compatible:
                max(compatible, key=lambda chain: len(chain[-1]["input"])).append(req)
            else:
                chains.append([req])

        for index, chain in enumerate(chains):
            key = base_key if len(chains) == 1 else f"{base_key}-{index + 1}"
            trajectories[key] = chain
    return trajectories

def est_tokens(obj):
    """Crude token estimate: serialized chars / 4. There is NO usage field in the
    export — every token number in this repo is an estimate. State that in your writeup."""
    return len(json.dumps(obj)) // 4 if not isinstance(obj, str) else len(obj) // 4

def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    reqs = [r for _, _, r in iter_requests(export)]
    models = Counter(r["model"] for r in reqs)
    print(f"requests={len(reqs)}  per-model request counts: {dict(models)}")
    groups = group_trajectories(reqs)
    sizes = sorted(len(v) for v in groups.values())
    print(f"reconstructed trajectories={len(groups)}  calls/trajectory min/median/max: "
          f"{sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}")
    mixed = [k for k, v in groups.items() if len({r['model'] for r in v}) > 1]
    print(f"trajectories with >1 model: {len(mixed)}"
          + ("  (premise says one model per trajectory — inspect these)" if mixed else "  (matches the one-model-per-trajectory premise)"))
    total = sum(est_tokens(r["input"]) for r in reqs)
    print(f"est. input tokens (chars/4, no usage in export): {total:,}")
    # spot-check one request for the expected fields
    r = reqs[0]
    missing = [k for k in ("model", "input", "tools") if k not in r]
    extra = [k for k in r if k not in ("model", "input", "tools")]
    print(f"schema check on first request: missing={missing or 'none'} extra={extra or 'none'}")

if __name__ == "__main__": main()
