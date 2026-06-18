from researchops_agent.agents.claim_extractor import extract_experiment_claims
from researchops_agent.agents.extractive import answer_from_evidence
from researchops_agent.agents.report_builder import build_research_report
from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document
from researchops_agent.retrieval.evidence import build_evidence_pack
from researchops_agent.retrieval.factory import build_retriever
from researchops_agent.runner.config_builder import suggest_experiment_config
from researchops_agent.schemas.answer import EvidencePack, ExtractiveAnswer
from researchops_agent.schemas.experiment import ExperimentClaim, ExperimentConfig, ResearchReport
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
