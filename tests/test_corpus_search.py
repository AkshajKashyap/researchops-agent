import pytest

from researchops_agent.corpus.search import build_index_for_corpus, search_corpus


def test_build_index_and_search_across_multiple_documents(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "forecast.md").write_text("Ridge and LSTM use RMSE.", encoding="utf-8")
    (docs / "graph.md").write_text("WL graph kernels use accuracy.", encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index_for_corpus(str(docs), str(index_dir))
    result = search_corpus(str(index_dir), "graph kernels accuracy", top_k=1)

    assert len(result.results) == 1
    assert result.results[0].source_path == str(docs / "graph.md")


def test_corpus_search_preserves_source_paths(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Ridge uses RMSE.", encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index_for_corpus(str(docs), str(index_dir))
    result = search_corpus(str(index_dir), "Ridge", top_k=1)

    assert result.results[0].source_path == str(docs / "a.md")


def test_corpus_search_missing_index_raises_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        search_corpus(str(tmp_path / "missing"), "query")
