import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass
class JsonlLogger:
    path: str

    def __post_init__(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
