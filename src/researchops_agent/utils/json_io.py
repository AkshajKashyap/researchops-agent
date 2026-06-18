import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_json(path: str, data: BaseModel | dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = data.model_dump() if isinstance(data, BaseModel) else data
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
