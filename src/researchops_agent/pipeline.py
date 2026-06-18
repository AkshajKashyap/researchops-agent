from time import perf_counter

from researchops_agent.agents.claim_extractor import extract_experiment_claims
from researchops_agent.agents.extractive import answer_from_evidence
from researchops_agent.agents.llm_grounded import answer_from_evidence_with_llm
from researchops_agent.agents.report_builder import build_research_report
from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document
from researchops_agent.observability.tracing import append_trace_jsonl, make_trace_id
from researchops_agent.retrieval.evidence import build_evidence_pack
from researchops_agent.retrieval.factory import build_retriever
from researchops_agent.runner.config_builder import suggest_experiment_config
from researchops_agent.schemas.answer import EvidencePack, ExtractiveAnswer
from researchops_agent.schemas.experiment import ExperimentClaim, ExperimentConfig, ResearchReport
from researchops_agent.schemas.llm import GroundedLLMAnswer, LLMProviderConfig, LLMTrace
from researchops_agent.schemas.retrieval import RetrievalResult


def load_chunk_retrieve(
    path: str, query: str, retriever_kind: str = "tfidf", top_k: int = 5
) -> RetrievalResult:
    pages = load_document(path)
    chunks = chunk_pages(pages)
    retriever = build_retriever(retriever_kind)
    retriever.fit(chunks)
    return retriever.search(query, top_k=top_k)


def build_evidence_from_document(
    path: str,
    query: str,
    retriever_kind: str = "tfidf",
    top_k: int = 5,
    min_score: float = 0.05,
) -> EvidencePack:
    retrieval = load_chunk_retrieve(
        path=path,
        query=query,
        retriever_kind=retriever_kind,
        top_k=top_k,
    )
    return build_evidence_pack(retrieval, min_score=min_score)


def ask_document(
    path: str, query: str, retriever_kind: str = "tfidf", top_k: int = 5
) -> ExtractiveAnswer:
    evidence = build_evidence_from_document(path, query, retriever_kind, top_k)
    return answer_from_evidence(evidence)


def extract_claims_from_document(
    path: str, query: str, retriever_kind: str = "tfidf", top_k: int = 5
) -> list[ExperimentClaim]:
    evidence = build_evidence_from_document(path, query, retriever_kind, top_k)
    return extract_experiment_claims(evidence)


def build_report_from_document(
    path: str, query: str, retriever_kind: str = "tfidf", top_k: int = 5
) -> ResearchReport:
    answer = ask_document(path, query, retriever_kind, top_k)
    claims = extract_claims_from_document(path, query, retriever_kind, top_k)
    return build_research_report(query, answer, claims)


def suggest_config_from_document(
    path: str, query: str, retriever_kind: str = "tfidf", top_k: int = 5
) -> ExperimentConfig | None:
    claims = extract_claims_from_document(path, query, retriever_kind, top_k)
    return suggest_experiment_config(claims)


def llm_ask_document(
    path: str,
    query: str,
    retriever_kind: str = "tfidf",
    top_k: int = 5,
    provider: str = "fake",
    model: str | None = None,
    trace_path: str | None = None,
) -> GroundedLLMAnswer:
    trace_id = make_trace_id()
    start = perf_counter()
    evidence = build_evidence_from_document(path, query, retriever_kind, top_k)
    retrieved_citations = [item.citation for item in evidence.items]
    provider_config = LLMProviderConfig(provider=provider, model=model)

    try:
        answer = answer_from_evidence_with_llm(evidence, provider_config)
    except Exception as exc:
        if trace_path:
            append_trace_jsonl(
                trace_path,
                LLMTrace(
                    trace_id=trace_id,
                    provider=provider,
                    model=model,
                    query=query,
                    retrieved_citations=retrieved_citations,
                    used_citations=[],
                    abstained=True,
                    latency_ms=(perf_counter() - start) * 1000,
                    error=str(exc),
                ),
            )
        raise

    if trace_path:
        append_trace_jsonl(
            trace_path,
            LLMTrace(
                trace_id=trace_id,
                provider=answer.provider,
                model=answer.model,
                query=query,
                retrieved_citations=retrieved_citations,
                used_citations=answer.citations,
                abstained=answer.abstained,
                latency_ms=(perf_counter() - start) * 1000,
            ),
        )

    return answer


def llm_report_from_document(
    path: str,
    query: str,
    retriever_kind: str = "tfidf",
    top_k: int = 5,
    provider: str = "fake",
    model: str | None = None,
    trace_path: str | None = None,
) -> ResearchReport:
    answer = llm_ask_document(
        path=path,
        query=query,
        retriever_kind=retriever_kind,
        top_k=top_k,
        provider=provider,
        model=model,
        trace_path=trace_path,
    )
    evidence = build_evidence_from_document(path, query, retriever_kind, top_k)
    claims = [] if answer.abstained else extract_experiment_claims(evidence)
    suggested_config = None if answer.abstained else suggest_experiment_config(claims)
    return ResearchReport(
        query=query,
        answer=answer.answer,
        citations=answer.citations,
        claims=claims,
        suggested_config=suggested_config,
        abstained=answer.abstained,
        reason=answer.reason,
    )
