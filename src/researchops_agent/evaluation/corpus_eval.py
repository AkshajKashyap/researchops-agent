from researchops_agent.corpus.search import search_corpus
from researchops_agent.retrieval.citations import format_citation
from researchops_agent.schemas.evaluation import RetrievalEvalCase, RetrievalEvalResult


def _matched_substrings(expected: list[str], texts: list[str]) -> list[str]:
    lowered_texts = [text.lower() for text in texts]
    return [
        substring
        for substring in expected
        if any(substring.lower() in text for text in lowered_texts)
    ]


def evaluate_corpus_retrieval(
    index_dir: str,
    cases: list[RetrievalEvalCase],
    top_k: int = 3,
    retriever_kind: str | None = None,
) -> list[RetrievalEvalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results: list[RetrievalEvalResult] = []
    for case in cases:
        corpus_result = search_corpus(
            index_dir,
            case.query,
            top_k=top_k,
            retriever_kind=retriever_kind,
        )
        retrieved_texts = [chunk.text for chunk in corpus_result.results]
        matches = _matched_substrings(case.expected_substrings, retrieved_texts)
        results.append(
            RetrievalEvalResult(
                case_id=case.case_id,
                query=case.query,
                top_k=top_k,
                hit=bool(matches),
                matched_substrings=matches,
                top_citations=[format_citation(result) for result in corpus_result.results],
                top_scores=[result.score for result in corpus_result.results],
            )
        )
    return results
