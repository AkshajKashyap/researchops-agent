import json

from researchops_agent.evaluation.load_cases import load_answer_cases, load_retrieval_cases


def test_loads_retrieval_cases_from_temp_json(tmp_path) -> None:
    path = tmp_path / "retrieval_cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "document_path": "doc.md",
                    "query": "What models?",
                    "expected_substrings": ["Ridge"],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_retrieval_cases(str(path))

    assert len(cases) == 1
    assert cases[0].case_id == "case-1"
    assert cases[0].expected_substrings == ["Ridge"]


def test_loads_answer_cases_from_temp_json(tmp_path) -> None:
    path = tmp_path / "answer_cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "document_path": "doc.md",
                    "query": "What metrics?",
                    "expected_answer_substrings": ["RMSE"],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_answer_cases(str(path))

    assert len(cases) == 1
    assert cases[0].case_id == "case-1"
    assert cases[0].expected_answer_substrings == ["RMSE"]
