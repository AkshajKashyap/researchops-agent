from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from researchops_agent.corpus.search import build_index_for_corpus, search_corpus
from researchops_agent.evaluation.answer_eval import evaluate_answers
from researchops_agent.evaluation.load_cases import load_answer_cases, load_retrieval_cases
from researchops_agent.evaluation.report import build_evaluation_report
from researchops_agent.evaluation.retrieval_eval import evaluate_retrieval
from researchops_agent.pipeline import (
    ask_document,
    ask_corpus,
    build_report_from_corpus,
    build_report_from_document,
    extract_claims_from_document,
    load_chunk_retrieve,
    llm_ask_corpus,
    llm_ask_document,
    llm_report_from_corpus,
    llm_report_from_document,
    suggest_config_from_document,
)
from researchops_agent.runner.experiment_runner import run_experiment_config
from researchops_agent.schemas.api import (
    CorpusIndexRequest,
    CorpusQueryRequest,
    DocumentQueryRequest,
    EvalRequest,
    LLMCorpusQueryRequest,
    LLMDocumentQueryRequest,
    RunConfigRequest,
    WorkflowRequest,
)
from researchops_agent.schemas.corpus import CorpusIndexMetadata, CorpusSearchResult
from researchops_agent.schemas.evaluation import EvaluationReport
from researchops_agent.schemas.experiment import ExperimentClaim, ExperimentConfig, ResearchReport
from researchops_agent.schemas.llm import GroundedLLMAnswer
from researchops_agent.schemas.retrieval import RetrievalResult
from researchops_agent.schemas.run import ExperimentRunResult
from researchops_agent.schemas.answer import ExtractiveAnswer
from researchops_agent.schemas.workflow import WorkflowOptions, WorkflowResult
from researchops_agent.utils.yaml_io import read_yaml
from researchops_agent.workflows.orchestrator import run_research_workflow

app = FastAPI(title="ResearchOps Agent")


def _as_bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(request: DocumentQueryRequest) -> RetrievalResult:
    try:
        return load_chunk_retrieve(
            request.path,
            request.query,
            retriever_kind=request.retriever,
            top_k=request.top_k,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/ask")
def ask(request: DocumentQueryRequest) -> ExtractiveAnswer:
    try:
        return ask_document(
            request.path,
            request.query,
            retriever_kind=request.retriever,
            top_k=request.top_k,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/claims")
def claims(request: DocumentQueryRequest) -> list[ExperimentClaim]:
    try:
        return extract_claims_from_document(
            request.path,
            request.query,
            retriever_kind=request.retriever,
            top_k=request.top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/report")
def report(request: DocumentQueryRequest) -> ResearchReport:
    try:
        return build_report_from_document(
            request.path,
            request.query,
            retriever_kind=request.retriever,
            top_k=request.top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/suggest-config")
def suggest_config(request: DocumentQueryRequest) -> ExperimentConfig | dict[str, str]:
    try:
        config = suggest_config_from_document(
            request.path,
            request.query,
            retriever_kind=request.retriever,
            top_k=request.top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc

    if config is None:
        return {"message": "No experiment config could be suggested from retrieved evidence."}
    return config


@app.post("/run-config")
def run_config(request: RunConfigRequest) -> ExperimentRunResult:
    try:
        config = ExperimentConfig.model_validate(read_yaml(request.config_path))
        return run_experiment_config(config, out_dir=request.out_dir, seed=request.seed)
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/eval")
def eval_endpoint(request: EvalRequest) -> EvaluationReport:
    try:
        retrieval_cases = load_retrieval_cases(request.retrieval_cases)
        answer_cases = load_answer_cases(request.answer_cases)
        retrieval_results = evaluate_retrieval(
            retrieval_cases,
            top_k=request.top_k,
            retriever_kind=request.retriever,
        )
        answer_results = evaluate_answers(answer_cases, retriever_kind=request.retriever)
        return build_evaluation_report(retrieval_results, answer_results)
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/llm/ask")
def llm_ask(request: LLMDocumentQueryRequest) -> GroundedLLMAnswer:
    try:
        return llm_ask_document(
            request.path,
            request.query,
            retriever_kind=request.retriever,
            top_k=request.top_k,
            provider=request.provider,
            model=request.model,
            trace_path=request.trace_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/llm/report")
def llm_report(request: LLMDocumentQueryRequest) -> ResearchReport:
    try:
        return llm_report_from_document(
            request.path,
            request.query,
            retriever_kind=request.retriever,
            top_k=request.top_k,
            provider=request.provider,
            model=request.model,
            trace_path=request.trace_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/corpus/index")
def corpus_index(request: CorpusIndexRequest) -> CorpusIndexMetadata:
    try:
        return build_index_for_corpus(
            root_path=request.root_path,
            index_dir=request.index_dir,
            retriever_kind=request.retriever,
            recursive=request.recursive,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/corpus/search")
def corpus_search(request: CorpusQueryRequest) -> CorpusSearchResult:
    try:
        return search_corpus(
            request.index_dir,
            request.query,
            top_k=request.top_k,
            retriever_kind=request.retriever,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/corpus/ask")
def corpus_ask(request: CorpusQueryRequest) -> ExtractiveAnswer:
    try:
        return ask_corpus(
            request.index_dir,
            request.query,
            top_k=request.top_k,
            retriever_kind=request.retriever,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/corpus/report")
def corpus_report(request: CorpusQueryRequest) -> ResearchReport:
    try:
        return build_report_from_corpus(
            request.index_dir,
            request.query,
            top_k=request.top_k,
            retriever_kind=request.retriever,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/llm/corpus/ask")
def llm_corpus_ask(request: LLMCorpusQueryRequest) -> GroundedLLMAnswer:
    try:
        return llm_ask_corpus(
            request.index_dir,
            request.query,
            top_k=request.top_k,
            retriever_kind=request.retriever,
            provider=request.provider,
            model=request.model,
            trace_path=request.trace_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/llm/corpus/report")
def llm_corpus_report(request: LLMCorpusQueryRequest) -> ResearchReport:
    try:
        return llm_report_from_corpus(
            request.index_dir,
            request.query,
            top_k=request.top_k,
            retriever_kind=request.retriever,
            provider=request.provider,
            model=request.model,
            trace_path=request.trace_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise _as_bad_request(exc) from exc


@app.post("/workflow")
def workflow(request: WorkflowRequest) -> WorkflowResult:
    try:
        return run_research_workflow(
            index_dir=request.index_dir,
            query=request.query,
            options=WorkflowOptions(
                retriever=request.retriever,
                top_k=request.top_k,
                use_llm=request.use_llm,
                llm_provider=request.llm_provider,
                llm_model=request.llm_model,
                run_if_runnable=request.run_if_runnable,
                seed=request.seed,
            ),
            out_dir=request.out_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise _as_bad_request(exc) from exc
