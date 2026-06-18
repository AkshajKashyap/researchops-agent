from researchops_agent.schemas.experiment import ResearchReport
from researchops_agent.utils.json_io import read_json, write_json


def test_writes_pydantic_model_to_json(tmp_path) -> None:
    path = tmp_path / "reports" / "report.json"
    report = ResearchReport(query="query", answer="answer")

    write_json(str(path), report)

    assert path.exists()
    assert '"query": "query"' in path.read_text(encoding="utf-8")


def test_reads_json_back_as_dict(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text('{"answer": "local"}', encoding="utf-8")

    data = read_json(str(path))

    assert data == {"answer": "local"}


def test_creates_parent_directories(tmp_path) -> None:
    path = tmp_path / "nested" / "reports" / "report.json"

    write_json(str(path), {"ok": True})

    assert path.exists()
