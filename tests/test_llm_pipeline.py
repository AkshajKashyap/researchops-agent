import json

from researchops_agent.pipeline import llm_ask_document, llm_report_from_document


def test_llm_ask_document_works_with_fake_provider_on_temp_markdown(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge and LSTM are compared using RMSE.", encoding="utf-8")

    answer = llm_ask_document(str(document), "What models were compared?", provider="fake")

    assert answer.abstained is False
    assert answer.citations
    assert "Ridge" in answer.answer


def test_llm_report_from_document_returns_research_report(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("We compare Ridge using RMSE.", encoding="utf-8")

    report = llm_report_from_document(str(document), "Ridge RMSE")

    assert report.answer
    assert report.citations
    assert report.abstained is False


def test_llm_pipeline_writes_trace_jsonl(tmp_path) -> None:
    document = tmp_path / "note.md"
    trace_path = tmp_path / "traces" / "llm.jsonl"
    document.write_text("Ridge and LSTM are compared using RMSE.", encoding="utf-8")

    llm_ask_document(
        str(document),
        "What models were compared?",
        provider="fake",
        trace_path=str(trace_path),
    )

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    trace = json.loads(lines[0])
    assert trace["provider"] == "fake"
    assert trace["retrieved_citations"]
    assert "Ridge and LSTM" not in lines[0]
