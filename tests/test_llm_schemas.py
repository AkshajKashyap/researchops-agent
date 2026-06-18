from researchops_agent.schemas.llm import GroundedLLMAnswer, LLMProviderConfig


def test_llm_provider_config_defaults() -> None:
    config = LLMProviderConfig()

    assert config.provider == "fake"
    assert config.model is None
    assert config.temperature == 0.0
    assert config.max_output_tokens == 800


def test_grounded_llm_answer_defaults() -> None:
    answer = GroundedLLMAnswer(query="q", answer="a")

    assert answer.citations == []
    assert answer.abstained is False
    assert answer.provider == "unknown"
