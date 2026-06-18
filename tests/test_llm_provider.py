import pytest

from researchops_agent.llm.factory import build_llm_provider
from researchops_agent.llm.fake import FakeLLMProvider
from researchops_agent.llm.openai_provider import OpenAIProvider
from researchops_agent.schemas.answer import EvidenceItem, EvidencePack
from researchops_agent.schemas.llm import LLMProviderConfig


def _pack(items: list[EvidenceItem]) -> EvidencePack:
    return EvidencePack(query="What is compared?", items=items)


def _item() -> EvidenceItem:
    return EvidenceItem(
        chunk_id="chunk-0",
        citation="paper.md#chunk=0",
        score=0.9,
        text="Ridge and LSTM are compared using RMSE.",
    )


def test_fake_provider_returns_grounded_answer_with_evidence() -> None:
    provider = FakeLLMProvider()

    answer = provider.generate_grounded_answer(
        "What is compared?",
        _pack([_item()]),
        LLMProviderConfig(),
    )

    assert answer.abstained is False
    assert answer.citations == ["paper.md#chunk=0"]
    assert "Ridge" in answer.answer


def test_fake_provider_abstains_with_empty_evidence() -> None:
    provider = FakeLLMProvider()

    answer = provider.generate_grounded_answer(
        "What is compared?",
        _pack([]),
        LLMProviderConfig(),
    )

    assert answer.abstained is True
    assert answer.citations == []


def test_factory_builds_fake_provider() -> None:
    assert isinstance(build_llm_provider("fake"), FakeLLMProvider)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_llm_provider("unknown")


def test_openai_provider_missing_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="OpenAI model is required"):
        OpenAIProvider().generate_grounded_answer(
            "query",
            _pack([_item()]),
            LLMProviderConfig(provider="openai"),
        )
