from researchops_agent.retrieval.citations import format_citation
from researchops_agent.schemas.answer import EvidenceItem, EvidencePack
from researchops_agent.schemas.retrieval import RetrievalResult


def build_evidence_pack(result: RetrievalResult, min_score: float = 0.0) -> EvidencePack:
    if min_score < 0:
        raise ValueError("min_score must be non-negative")

    items = [
        EvidenceItem(
            chunk_id=chunk.chunk_id,
            citation=format_citation(chunk),
            score=chunk.score,
            text=chunk.text,
        )
        for chunk in result.results
        if chunk.score >= min_score
    ]

    return EvidencePack(query=result.query, items=items)
