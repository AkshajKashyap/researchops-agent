from researchops_agent.corpus.search import build_index_for_corpus
from researchops_agent.pipeline import (
    ask_corpus,
    build_report_from_corpus,
    llm_ask_corpus,
    llm_report_from_corpus,
)


def _index(tmp_path) -> str:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "forecast.md").write_text("Ridge and LSTM use RMSE.", encoding="utf-8")
    (docs / "graph.md").write_text("WL graph kernels use accuracy.", encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index_for_corpus(str(docs), str(index_dir))
    return str(index_dir)


def test_ask_corpus_works(tmp_path) -> None:
    answer = ask_corpus(_index(tmp_path), "Ridge RMSE")

    assert answer.abstained is False
    assert answer.citations


def test_llm_ask_corpus_works_with_fake_provider(tmp_path) -> None:
    answer = llm_ask_corpus(_index(tmp_path), "graph kernels accuracy", provider="fake")

    assert answer.abstained is False
    assert answer.citations


def test_build_report_from_corpus_works(tmp_path) -> None:
    report = build_report_from_corpus(_index(tmp_path), "Ridge RMSE")

    assert report.answer
    assert report.citations


def test_llm_report_from_corpus_works_with_fake_provider(tmp_path) -> None:
    report = llm_report_from_corpus(_index(tmp_path), "graph kernels accuracy", provider="fake")

    assert report.answer
    assert report.citations
