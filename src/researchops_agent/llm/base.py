from typing import Protocol

from researchops_agent.schemas.answer import EvidencePack
from researchops_agent.schemas.llm import GroundedLLMAnswer, LLMProviderConfig


class BaseLLMProvider(Protocol):
    def generate_grounded_answer(
        self,
        query: str,
        evidence: EvidencePack,
        config: LLMProviderConfig,
    ) -> GroundedLLMAnswer:
        ...
