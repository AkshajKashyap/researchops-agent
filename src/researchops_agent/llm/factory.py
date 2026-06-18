from researchops_agent.llm.base import BaseLLMProvider
from researchops_agent.llm.fake import FakeLLMProvider
from researchops_agent.llm.openai_provider import OpenAIProvider


def build_llm_provider(provider: str) -> BaseLLMProvider:
    normalized_provider = provider.strip().lower()
    if normalized_provider == "fake":
        return FakeLLMProvider()
    if normalized_provider == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unsupported LLM provider: {provider}")
