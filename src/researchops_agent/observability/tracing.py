import json
import uuid
from pathlib import Path

from researchops_agent.schemas.llm import LLMTrace


def make_trace_id() -> str:
    return uuid.uuid4().hex


def write_trace(path: str, trace: LLMTrace) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace.model_dump(), indent=2) + "\n", encoding="utf-8")


def append_trace_jsonl(path: str, trace: LLMTrace) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trace.model_dump()) + "\n")
