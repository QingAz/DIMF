from __future__ import annotations

from importlib.machinery import SourcelessFileLoader
from pathlib import Path


def load_pyc_into_globals(module_name: str, pyc_path: str | Path, target_globals: dict) -> None:
    path = Path(pyc_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached module: {path}")
    cached = SourcelessFileLoader(f"{module_name}.__cached__", str(path)).load_module()
    copied = {
        key: value
        for key, value in cached.__dict__.items()
        if key not in {"__name__", "__loader__", "__package__", "__spec__"}
    }
    target_globals.update(copied)
