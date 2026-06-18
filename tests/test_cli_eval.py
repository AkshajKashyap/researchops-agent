import json

from typer.testing import CliRunner

from researchops_agent.cli import app


def test_eval_command_works_against_temp_eval_files(tmp_path) -> None:
    document = tmp_path / "note.md"
    retrieval_cases = tmp_path / "retrieval_cases.json"
    answer_cases = tmp_path / "answer_cases.json"
    out_json = tmp_path / "reports" / "evaluation.json"
    out_md = tmp_path / "reports" / "evaluation.md"

    document.write_text(
        "We compare Ridge and LSTM using RMSE. "
        "The chronological split holds out the final week.",
        encoding="utf-8",
    )
    retrieval_cases.write_text(
        json.dumps(
            [
                {
                    "case_id": "retrieval",
                    "document_path": str(document),
                    "query": "Ridge RMSE",
                    "expected_substrings": ["Ridge and LSTM"],
                }
            ]
        ),
        encoding="utf-8",
    )
    answer_cases.write_text(
        json.dumps(
            [
                {
                    "case_id": "answer",
                    "document_path": str(document),
                    "query": "Ridge RMSE",
                    "expected_answer_substrings": ["Ridge"],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--retriever",
            "tfidf",
            "--retrieval-cases",
            str(retrieval_cases),
            "--answer-cases",
            str(answer_cases),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )

    assert result.exit_code == 0
    assert "Retrieval hit rate:" in result.output
    assert "Answer pass rate:" in result.output
    assert out_json.exists()
    assert out_md.exists()


def test_compare_retrievers_command_works_with_tfidf_on_temp_eval_files(tmp_path) -> None:
    document = tmp_path / "note.md"
    retrieval_cases = tmp_path / "retrieval_cases.json"
    answer_cases = tmp_path / "answer_cases.json"
    out_dir = tmp_path / "comparison"

    document.write_text("We compare Ridge and LSTM using RMSE.", encoding="utf-8")
    retrieval_cases.write_text(
        json.dumps(
            [
                {
                    "case_id": "retrieval",
                    "document_path": str(document),
                    "query": "Ridge RMSE",
                    "expected_substrings": ["Ridge"],
                }
            ]
        ),
        encoding="utf-8",
    )
    answer_cases.write_text(
        json.dumps(
            [
                {
                    "case_id": "answer",
                    "document_path": str(document),
                    "query": "Ridge RMSE",
                    "expected_answer_substrings": ["Ridge"],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "compare-retrievers",
            "--retrievers",
            "tfidf",
            "--retrieval-cases",
            str(retrieval_cases),
            "--answer-cases",
            str(answer_cases),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert "| tfidf |" in result.output
    assert (out_dir / "tfidf_evaluation.json").exists()
    assert (out_dir / "tfidf_evaluation.md").exists()
