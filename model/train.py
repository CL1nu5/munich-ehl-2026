#!/usr/bin/env python3
"""Training entry point.

The pipeline this used to inline now lives in the package around it:

* :mod:`model.data` builds and loads the leakage-safe splits
* :mod:`model.embed` chunks, deduplicates, encodes and caches the text
* :mod:`model.heads` fits the ridge / MLP heads
* :mod:`model.pipeline` runs the lot and scores the held-out split

Usage (from the repo root)::

    python model/train.py                      # static encoder, seconds
    python model/train.py --encoder minilm     # MiniLM, ~2 min on a cold cache
    python -m model --head ridge               # same thing through the package
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python model/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.pipeline import main

if __name__ == "__main__":
    main()
