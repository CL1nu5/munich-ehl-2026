"""Prompt-complexity model for the Viktor routing challenge.

Typical use::

    from model import fast_config, run_pipeline
    report = run_pipeline(fast_config())
"""
from .config import ENCODERS, SEGMENTS, TARGETS, PipelineConfig, accurate_config, fast_config
from .pipeline import run_pipeline

__all__ = [
    "ENCODERS",
    "SEGMENTS",
    "TARGETS",
    "PipelineConfig",
    "accurate_config",
    "fast_config",
    "run_pipeline",
]
