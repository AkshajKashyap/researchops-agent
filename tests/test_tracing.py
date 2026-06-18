import json

from researchops_agent.observability.tracing import append_trace_jsonl, write_trace
from researchops_agent.schemas.llm import LLMTrace


def _trace() -> LLMTrace:
    return LLMTrace(
        trace_id="trace-1",
        provider="fake",
        model=None,
        query="What is compared?",
        retrieved_citations=["paper.md#chunk=0"],
        used_citations=["paper.md#chunk=0"],
        abstained=False,
        latency_ms=1.2,
    )


def test_write_trace_json(tmp_path) -> None:
    path = tmp_path / "traces" / "trace.json"

    write_trace(str(path), _trace())

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["trace_id"] == "trace-1"


def test_append_trace_jsonl(tmp_path) -> None:
    path = tmp_path / "traces" / "trace.jsonl"

    append_trace_jsonl(str(path), _trace())
    append_trace_jsonl(str(path), _trace())

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["provider"] == "fake"


def test_trace_does_not_include_evidence_text(tmp_path) -> None:
    path = tmp_path / "trace.json"

    write_trace(str(path), _trace())

    assert "Evidence text" not in path.read_text(encoding="utf-8")
