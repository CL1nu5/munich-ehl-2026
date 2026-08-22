"""Dataset setup: build the leakage-safe splits if needed, then load them.

The heavy lifting lives in ``scripts/build_complexity_dataset.py``; this module
only decides whether a rebuild is required, loads the result, and re-checks the
invariants the builder claims to guarantee.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig

SPLITS = ("train", "validation", "test")


@dataclass
class Split:
    """One aligned split: predictor inputs, targets, and audit-only metadata."""

    name: str
    request_ids: list[str]
    inputs: list[dict]
    targets: list[dict]
    metadata: list[dict]

    def __len__(self) -> int:
        return len(self.request_ids)

    def column(self, field: str) -> list:
        return [row[field] for row in self.targets]


def _split_paths(config: PipelineConfig, split: str) -> dict[str, Path]:
    return {
        kind: config.dataset_dir / f"{split}_{kind}.jsonl"
        for kind in ("inputs", "targets", "metadata")
    }


def dataset_is_current(config: PipelineConfig) -> bool:
    """True when a manifest built with this config's split settings is on disk."""
    manifest_path = config.dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    for split in SPLITS:
        if not all(path.exists() for path in _split_paths(config, split).values()):
            return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    ratios = manifest.get("requested_split_ratios", {})
    return (
        manifest.get("split_seed") == config.seed
        and abs(ratios.get("train", -1) - config.train_ratio) < 1e-9
        and abs(ratios.get("validation", -1) - config.validation_ratio) < 1e-9
        and abs(ratios.get("test", -1) - config.test_ratio) < 1e-9
    )


def ensure_dataset(config: PipelineConfig, *, verbose: bool = True) -> dict:
    """Build the split files unless a matching build already exists.

    Returns the manifest.
    """
    if config.rebuild_dataset or not dataset_is_current(config):
        if not config.export_dir.exists():
            raise FileNotFoundError(
                f"Export directory {config.export_dir} is missing. Extract the challenge "
                "archive into it, or run scripts/make_synthetic_sample.py for a stand-in."
            )
        command = [
            sys.executable,
            str(config.root / "scripts" / "build_complexity_dataset.py"),
            str(config.export_dir),
            "--output-dir", str(config.dataset_dir),
            "--seed", str(config.seed),
            "--train-ratio", str(config.train_ratio),
            "--validation-ratio", str(config.validation_ratio),
            "--test-ratio", str(config.test_ratio),
        ]
        if verbose:
            print(f"Building dataset -> {config.dataset_dir}")
        subprocess.run(command, cwd=config.root, check=True,
                       stdout=None if verbose else subprocess.DEVNULL)
    elif verbose:
        print(f"Reusing dataset at {config.dataset_dir}")
    return json.loads((config.dataset_dir / "manifest.json").read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_splits(config: PipelineConfig) -> dict[str, Split]:
    """Load all three splits, asserting the files are row-aligned by request_id."""
    splits: dict[str, Split] = {}
    for name in SPLITS:
        paths = _split_paths(config, name)
        inputs = read_jsonl(paths["inputs"])
        targets = read_jsonl(paths["targets"])
        metadata = read_jsonl(paths["metadata"])
        ids = [row["request_id"] for row in inputs]
        if ids != [row["request_id"] for row in targets] or ids != [
            row["request_id"] for row in metadata
        ]:
            raise ValueError(f"{name}: inputs/targets/metadata are not row-aligned")
        splits[name] = Split(name, ids, inputs, targets, metadata)
    return splits


def check_leakage(splits: dict[str, Split], manifest: dict) -> dict:
    """Re-derive the builder's leakage guarantees instead of trusting the manifest."""
    seen: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    trajectory_splits: dict[str, set[str]] = defaultdict(set)

    for split in splits.values():
        duplicates = seen & set(split.request_ids)
        if duplicates:
            raise AssertionError(f"{split.name}: {len(duplicates)} request ids appear twice")
        seen.update(split.request_ids)
        for row in split.metadata:
            group_splits[row["split_group_id"]].add(split.name)
            trajectory_splits[row["trajectory_id"]].add(split.name)

    leaked_groups = [g for g, s in group_splits.items() if len(s) > 1]
    leaked_trajectories = [t for t, s in trajectory_splits.items() if len(s) > 1]
    if leaked_groups or leaked_trajectories:
        raise AssertionError(
            f"Split leakage: {len(leaked_groups)} prompt groups and "
            f"{len(leaked_trajectories)} trajectories span more than one split"
        )
    if len(seen) != manifest["request_count"]:
        raise AssertionError(
            f"Loaded {len(seen)} requests but the manifest claims {manifest['request_count']}"
        )
    return {
        "requests": len(seen),
        "split_groups": len(group_splits),
        "trajectories": len(trajectory_splits),
        "leaked_split_groups": 0,
        "leaked_trajectories": 0,
    }
