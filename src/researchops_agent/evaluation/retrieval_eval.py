from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document
from researchops_agent.retrieval.citations import format_citation
from researchops_agent.retrieval.tfidf import TfidfRetriever
from researchops_agent.schemas.evaluation import RetrievalEvalCase, RetrievalEvalResult


def _matched_substrings(expected: list[str], texts: list[str]) -> list[str]:
    lowered_texts = [text.lower() for text in texts]
    return [
        substring
        for substring in expected
        if any(substring.lower() in text for text in lowered_texts)
    ]


def evaluate_retrieval(
    cases: list[RetrievalEvalCase], top_k: int = 3
) -> list[RetrievalEvalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results: list[RetrievalEvalResult] = []
    for case in cases:
        pages = load_document(case.document_path)
        chunks = chunk_pages(pages)
        retriever = TfidfRetriever()
        retriever.fit(chunks)
        retrieval = retriever.search(case.query, top_k=top_k)

        retrieved_texts = [chunk.text for chunk in retrieval.results]
        matches = _matched_substrings(case.expected_substrings, retrieved_texts)

        results.append(
            RetrievalEvalResult(
                case_id=case.case_id,
                query=case.query,
                top_k=top_k,
                hit=bool(matches),
                matched_substrings=matches,
                top_citations=[format_citation(chunk) for chunk in retrieval.results],
                top_scores=[chunk.score for chunk in retrieval.results],
            )
        )

    return results
