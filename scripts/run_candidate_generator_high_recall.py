#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess._pyc_restore import load_pyc_into_globals


load_pyc_into_globals(__name__, Path(__file__).with_name("__pycache__") / "run_candidate_generator_high_recall.cpython-310.pyc", globals())

if __name__ == "__main__":
    main()
