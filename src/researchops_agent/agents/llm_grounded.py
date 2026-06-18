from researchops_agent.llm.factory import build_llm_provider
from researchops_agent.llm.validation import validate_grounded_answer
from researchops_agent.schemas.answer import EvidencePack
from researchops_agent.schemas.llm import GroundedLLMAnswer, LLMProviderConfig


def answer_from_evidence_with_llm(
    pack: EvidencePack,
    provider_config: LLMProviderConfig,
) -> GroundedLLMAnswer:
    if not pack.items:
        return GroundedLLMAnswer(
            query=pack.query,
            answer="I do not have enough retrieved evidence to answer this question.",
            citations=[],
            abstained=True,
            reason="Insufficient retrieved evidence.",
            provider=provider_config.provider,
            model=provider_config.model,
        )

    provider = build_llm_provider(provider_config.provider)
    answer = provider.generate_grounded_answer(pack.query, pack, provider_config)
    return validate_grounded_answer(answer, pack)
