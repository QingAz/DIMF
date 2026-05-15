from __future__ import annotations

from pathlib import Path

from ._pyc_restore import load_pyc_into_globals


load_pyc_into_globals(__name__, Path(__file__).with_name("__pycache__") / "light_veto_filter.cpython-310.pyc", globals())
