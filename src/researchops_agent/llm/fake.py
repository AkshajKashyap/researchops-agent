from researchops_agent.schemas.answer import EvidencePack
from researchops_agent.schemas.llm import GroundedLLMAnswer, LLMProviderConfig

ABSTAINED_LLM_ANSWER = "I do not have enough retrieved evidence to answer this question."


class FakeLLMProvider:
    def generate_grounded_answer(
        self,
        query: str,
        evidence: EvidencePack,
        config: LLMProviderConfig,
    ) -> GroundedLLMAnswer:
        if not evidence.items:
            return GroundedLLMAnswer(
                query=query,
                answer=ABSTAINED_LLM_ANSWER,
                citations=[],
                abstained=True,
                reason="Insufficient retrieved evidence.",
                provider="fake",
                model=config.model,
            )

        top_item = evidence.items[0]
        answer_text = " ".join(top_item.text.split())
        if len(answer_text) > 500:
            answer_text = answer_text[:500].rstrip()

        return GroundedLLMAnswer(
            query=query,
            answer=answer_text,
            citations=[top_item.citation],
            abstained=False,
            provider="fake",
            model=config.model,
        )
