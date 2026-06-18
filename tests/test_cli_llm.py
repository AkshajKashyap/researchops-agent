import json

from typer.testing import CliRunner

from researchops_agent.cli import app


def test_llm_ask_command_works_with_fake_provider_on_temp_markdown(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge and LSTM are compared using RMSE.", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "llm-ask",
            str(document),
            "What models were compared?",
            "--provider",
            "fake",
            "--trace",
            str(tmp_path / "trace.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert "Answer:" in result.output
    assert "Citations:" in result.output


def test_llm_report_command_works_with_fake_provider_and_writes_json(tmp_path) -> None:
    document = tmp_path / "note.md"
    output = tmp_path / "llm_report.json"
    document.write_text("We compare Ridge using RMSE.", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "llm-report",
            str(document),
            "What experiment is described?",
            "--provider",
            "fake",
            "--out",
            str(output),
            "--trace",
            str(tmp_path / "trace.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["answer"]


def test_eval_llm_command_works_with_fake_provider_and_temp_cases(tmp_path) -> None:
    document = tmp_path / "note.md"
    cases = tmp_path / "answer_cases.json"
    out_json = tmp_path / "llm_eval.json"
    out_md = tmp_path / "llm_eval.md"
    document.write_text("Ridge and LSTM are compared using RMSE.", encoding="utf-8")
    cases.write_text(
        json.dumps(
            [
                {
                    "case_id": "answer",
                    "document_path": str(document),
                    "query": "What models were compared?",
                    "expected_answer_substrings": ["Ridge"],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-llm",
            "--answer-cases",
            str(cases),
            "--provider",
            "fake",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )

    assert result.exit_code == 0
    assert "LLM answer pass rate:" in result.output
    assert out_json.exists()
    assert out_md.exists()
