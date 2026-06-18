import re

from researchops_agent.schemas.answer import EvidencePack, ExtractiveAnswer

ABSTAINED_ANSWER = "I do not have enough retrieved evidence to answer this question."


def _terms(text: str) -> set[str]:
    return {term.lower() for term in re.findall(r"\b\w+\b", text)}


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _preview(text: str, max_chars: int = 300) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip()


def answer_from_evidence(pack: EvidencePack, min_items: int = 1) -> ExtractiveAnswer:
    if min_items <= 0:
        raise ValueError("min_items must be positive")

    if len(pack.items) < min_items:
        return ExtractiveAnswer(
            query=pack.query,
            answer=ABSTAINED_ANSWER,
            citations=[],
            abstained=True,
            reason="Insufficient retrieved evidence.",
        )

    top_item = pack.items[0]
    query_terms = _terms(pack.query)
    candidates = _sentences(top_item.text)

    if candidates and query_terms:
        answer = max(
            candidates,
            key=lambda sentence: (len(_terms(sentence) & query_terms), -len(sentence)),
        )
    elif candidates:
        answer = candidates[0]
    else:
        answer = _preview(top_item.text)

    return ExtractiveAnswer(
        query=pack.query,
        answer=answer,
        citations=[top_item.citation],
        abstained=False,
    )
