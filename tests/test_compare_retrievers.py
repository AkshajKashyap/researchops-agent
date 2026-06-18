import pytest

from researchops_agent.evaluation.compare import compare_retrievers
from researchops_agent.retrieval import factory as retriever_factory
from researchops_agent.retrieval.embedding import EmbeddingRetriever
from researchops_agent.schemas.evaluation import AnswerEvalCase, RetrievalEvalCase


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        terms = ["ridge", "rmse"]
        return [[float(text.lower().count(term)) for term in terms] for text in texts]


def test_compare_retrievers_returns_reports_for_multiple_retriever_kinds(
    tmp_path, monkeypatch
) -> None:
    document = tmp_path / "note.md"
    document.write_text("We compare Ridge using RMSE.", encoding="utf-8")
    retrieval_case = RetrievalEvalCase(
        case_id="retrieval",
        document_path=str(document),
        query="Ridge RMSE",
        expected_substrings=["Ridge"],
    )
    answer_case = AnswerEvalCase(
        case_id="answer",
        document_path=str(document),
        query="Ridge RMSE",
        expected_answer_substrings=["Ridge"],
    )

    monkeypatch.setattr(
        retriever_factory,
        "EmbeddingRetriever",
        lambda: EmbeddingRetriever(model=FakeEmbeddingModel()),
    )

    reports = compare_retrievers(
        [retrieval_case], [answer_case], ["tfidf", "embedding"], top_k=1
    )

    assert set(reports) == {"tfidf", "embedding"}
    assert reports["tfidf"].summary.retrieval_cases == 1
    assert reports["embedding"].summary.answer_cases == 1


def test_compare_retrievers_rejects_empty_retriever_list() -> None:
    with pytest.raises(ValueError, match="retriever_kinds must not be empty"):
        compare_retrievers([], [], [])
