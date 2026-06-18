import pytest

from researchops_agent.retrieval.embedding import EmbeddingRetriever
from researchops_agent.retrieval.factory import build_retriever
from researchops_agent.retrieval.tfidf import TfidfRetriever


def test_factory_builds_tfidf_retriever() -> None:
    assert isinstance(build_retriever("tfidf"), TfidfRetriever)


def test_factory_builds_embedding_retriever_without_loading_model() -> None:
    assert isinstance(build_retriever("embedding"), EmbeddingRetriever)


def test_factory_rejects_unknown_retriever_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported retriever kind"):
        build_retriever("unknown")
