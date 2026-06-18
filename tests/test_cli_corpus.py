import json

from typer.testing import CliRunner

from researchops_agent.cli import app


def _docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "forecast.md").write_text("Ridge and LSTM use RMSE.", encoding="utf-8")
    (docs / "graph.md").write_text("WL graph kernels use accuracy.", encoding="utf-8")
    return docs


def test_index_corpus_command_works_on_temp_docs(tmp_path) -> None:
    index_dir = tmp_path / "index"

    result = CliRunner().invoke(
        app,
        ["index-corpus", str(_docs(tmp_path)), "--index-dir", str(index_dir)],
    )

    assert result.exit_code == 0
    assert "Corpus ID:" in result.output
    assert (index_dir / "manifest.json").exists()


def test_search_corpus_command_works(tmp_path) -> None:
    docs = _docs(tmp_path)
    index_dir = tmp_path / "index"
    CliRunner().invoke(app, ["index-corpus", str(docs), "--index-dir", str(index_dir)])

    result = CliRunner().invoke(
        app,
        ["search-corpus", str(index_dir), "graph kernels accuracy", "--top-k", "1"],
    )

    assert result.exit_code == 0
    assert "Rank: 1" in result.output
    assert "graph.md" in result.output


def test_ask_corpus_command_works(tmp_path) -> None:
    docs = _docs(tmp_path)
    index_dir = tmp_path / "index"
    CliRunner().invoke(app, ["index-corpus", str(docs), "--index-dir", str(index_dir)])

    result = CliRunner().invoke(app, ["ask-corpus", str(index_dir), "Ridge RMSE"])

    assert result.exit_code == 0
    assert "Answer:" in result.output


def test_llm_ask_corpus_command_works_with_fake_provider(tmp_path) -> None:
    docs = _docs(tmp_path)
    index_dir = tmp_path / "index"
    CliRunner().invoke(app, ["index-corpus", str(docs), "--index-dir", str(index_dir)])

    result = CliRunner().invoke(
        app,
        [
            "llm-ask-corpus",
            str(index_dir),
            "graph kernels accuracy",
            "--provider",
            "fake",
            "--trace",
            str(tmp_path / "trace.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert "Citations:" in result.output


def test_eval_corpus_command_works(tmp_path) -> None:
    docs = _docs(tmp_path)
    index_dir = tmp_path / "index"
    cases = tmp_path / "cases.json"
    out_json = tmp_path / "corpus_eval.json"
    out_md = tmp_path / "corpus_eval.md"
    CliRunner().invoke(app, ["index-corpus", str(docs), "--index-dir", str(index_dir)])
    cases.write_text(
        json.dumps(
            [
                {
                    "case_id": "graph",
                    "document_path": "ignored.md",
                    "query": "graph kernels accuracy",
                    "expected_substrings": ["WL graph kernels"],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-corpus",
            "--index-dir",
            str(index_dir),
            "--cases",
            str(cases),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )

    assert result.exit_code == 0
    assert "Corpus retrieval hit rate:" in result.output
    assert out_json.exists()
    assert out_md.exists()
