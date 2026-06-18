import pytest

from researchops_agent.evaluation.retrieval_eval import evaluate_retrieval
from researchops_agent.schemas.evaluation import RetrievalEvalCase


def _document(tmp_path, text: str) -> str:
    path = tmp_path / "note.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_retrieval_eval_hits_when_expected_substring_is_retrieved(tmp_path) -> None:
    document_path = _document(tmp_path, "We compare Ridge and LSTM using RMSE.")
    case = RetrievalEvalCase(
        case_id="hit",
        document_path=document_path,
        query="Ridge RMSE",
        expected_substrings=["Ridge and LSTM"],
    )

    results = evaluate_retrieval([case], top_k=1, retriever_kind="tfidf")

    assert results[0].hit is True
    assert results[0].matched_substrings == ["Ridge and LSTM"]
    assert results[0].top_citations == [f"{document_path}#chunk=0"]


def test_retrieval_eval_misses_when_expected_substring_is_absent(tmp_path) -> None:
    document_path = _document(tmp_path, "We compare Ridge using RMSE.")
    case = RetrievalEvalCase(
        case_id="miss",
        document_path=document_path,
        query="Ridge RMSE",
        expected_substrings=["Transformer"],
    )

    results = evaluate_retrieval([case], top_k=1)

    assert results[0].hit is False
    assert results[0].matched_substrings == []


def test_retrieval_eval_rejects_invalid_top_k(tmp_path) -> None:
    document_path = _document(tmp_path, "We compare Ridge using RMSE.")
    case = RetrievalEvalCase(
        case_id="invalid",
        document_path=document_path,
        query="Ridge",
        expected_substrings=["Ridge"],
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        evaluate_retrieval([case], top_k=0)


def test_retrieval_eval_rejects_invalid_retriever_kind(tmp_path) -> None:
    document_path = _document(tmp_path, "We compare Ridge using RMSE.")
    case = RetrievalEvalCase(
        case_id="invalid",
        document_path=document_path,
        query="Ridge",
        expected_substrings=["Ridge"],
    )

    with pytest.raises(ValueError, match="Unsupported retriever kind"):
        evaluate_retrieval([case], retriever_kind="unknown")
