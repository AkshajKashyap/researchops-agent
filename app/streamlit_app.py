import pandas as pd
import streamlit as st

from researchops_agent.corpus.search import build_index_for_corpus, search_corpus
from researchops_agent.evaluation.answer_eval import evaluate_answers
from researchops_agent.evaluation.load_cases import load_answer_cases, load_retrieval_cases
from researchops_agent.evaluation.report import build_evaluation_report, format_evaluation_markdown
from researchops_agent.evaluation.retrieval_eval import evaluate_retrieval
from researchops_agent.pipeline import (
    ask_document,
    ask_corpus,
    build_report_from_document,
    extract_claims_from_document,
    load_chunk_retrieve,
    llm_ask_document,
    llm_ask_corpus,
    llm_report_from_document,
    suggest_config_from_document,
)
from researchops_agent.runner.experiment_runner import run_experiment_config
from researchops_agent.runner.registry import summarize_runs
from researchops_agent.schemas.experiment import ExperimentConfig
from researchops_agent.schemas.workflow import WorkflowOptions
from researchops_agent.utils.yaml_io import read_yaml
from researchops_agent.workflows.orchestrator import run_research_workflow

DEFAULT_DOC = "examples/docs/time_series_note.md"
DEFAULT_QUERY = "What experiment is described?"
DEFAULT_CONFIG = "configs/time_series_demo.yaml"


def _query_controls(key_prefix: str) -> tuple[str, str, str, int]:
    path = st.text_input("Document path", DEFAULT_DOC, key=f"{key_prefix}_path")
    query = st.text_input("Query", DEFAULT_QUERY, key=f"{key_prefix}_query")
    retriever = st.selectbox("Retriever", ["tfidf", "embedding"], key=f"{key_prefix}_retriever")
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, key=f"{key_prefix}_top_k")
    return path, query, retriever, int(top_k)


st.set_page_config(page_title="ResearchOps Agent", layout="wide")
st.title("ResearchOps Agent")
st.warning(
    "This dashboard wraps deterministic local functionality. Embedding retrieval may require "
    "a first-time local model download, and the bounded runner supports only limited configs."
)

tabs = st.tabs(
    [
        "Overview",
        "Ask Document",
        "Extract Claims",
        "Build Report",
        "Suggest Config",
        "Run Config",
        "Evaluation",
        "LLM Grounded Answer",
        "Corpus",
        "Workflow",
    ]
)

with tabs[0]:
    st.subheader("Health / Overview")
    st.json({"status": "ok", "mode": "local deterministic"})
    runs = summarize_runs()
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True)
    else:
        st.info("No run directories found under reports/runs.")

with tabs[1]:
    st.subheader("Ask Document")
    path, query, retriever, top_k = _query_controls("ask")
    if st.button("Ask", key="ask_button"):
        answer = ask_document(path, query, retriever_kind=retriever, top_k=top_k)
        st.json(answer.model_dump())

with tabs[2]:
    st.subheader("Extract Claims")
    path, query, retriever, top_k = _query_controls("claims")
    if st.button("Extract Claims", key="claims_button"):
        claims = extract_claims_from_document(path, query, retriever_kind=retriever, top_k=top_k)
        st.json([claim.model_dump() for claim in claims])

with tabs[3]:
    st.subheader("Build Report")
    path, query, retriever, top_k = _query_controls("report")
    if st.button("Build Report", key="report_button"):
        report = build_report_from_document(path, query, retriever_kind=retriever, top_k=top_k)
        st.json(report.model_dump())

with tabs[4]:
    st.subheader("Suggest Config")
    path, query, retriever, top_k = _query_controls("suggest")
    if st.button("Suggest Config", key="suggest_button"):
        config = suggest_config_from_document(path, query, retriever_kind=retriever, top_k=top_k)
        if config is None:
            st.info("No config could be suggested from retrieved evidence.")
        else:
            st.json(config.model_dump())

with tabs[5]:
    st.subheader("Run Config")
    config_path = st.text_input("Config path", DEFAULT_CONFIG)
    out_dir = st.text_input("Output directory", "reports/runs")
    seed = st.number_input("Seed", value=42, step=1)
    if st.button("Run Config"):
        config = ExperimentConfig.model_validate(read_yaml(config_path))
        result = run_experiment_config(config, out_dir=out_dir, seed=int(seed))
        st.json(result.model_dump())

with tabs[6]:
    st.subheader("Evaluation")
    retrieval_cases = st.text_input("Retrieval cases", "examples/eval/retrieval_cases.json")
    answer_cases = st.text_input("Answer cases", "examples/eval/answer_cases.json")
    retriever = st.selectbox("Evaluation retriever", ["tfidf", "embedding"])
    top_k = st.number_input("Evaluation Top K", min_value=1, max_value=20, value=3)
    if st.button("Run Evaluation"):
        retrieval_results = evaluate_retrieval(
            load_retrieval_cases(retrieval_cases),
            top_k=int(top_k),
            retriever_kind=retriever,
        )
        answer_results = evaluate_answers(load_answer_cases(answer_cases), retriever_kind=retriever)
        report = build_evaluation_report(retrieval_results, answer_results)
        st.json(report.model_dump())
        st.markdown(format_evaluation_markdown(report))

with tabs[7]:
    st.subheader("LLM Grounded Answer")
    st.info(
        "The fake provider is deterministic and offline. The OpenAI provider requires "
        "environment configuration and must still cite retrieved evidence."
    )
    path, query, retriever, top_k = _query_controls("llm")
    provider = st.selectbox("LLM provider", ["fake", "openai"], key="llm_provider")
    model = st.text_input("Model", "", key="llm_model")
    trace_path = st.text_input(
        "Trace path",
        "reports/traces/llm_traces.jsonl",
        key="llm_trace",
    )
    if st.button("LLM Ask", key="llm_ask_button"):
        answer = llm_ask_document(
            path,
            query,
            retriever_kind=retriever,
            top_k=top_k,
            provider=provider,
            model=model or None,
            trace_path=trace_path or None,
        )
        st.json(answer.model_dump())
    if st.button("LLM Report", key="llm_report_button"):
        report = llm_report_from_document(
            path,
            query,
            retriever_kind=retriever,
            top_k=top_k,
            provider=provider,
            model=model or None,
            trace_path=trace_path or None,
        )
        st.json(report.model_dump())

with tabs[8]:
    st.subheader("Corpus")
    root_path = st.text_input("Corpus root path", "examples/docs", key="corpus_root")
    index_dir = st.text_input("Index directory", "data/indexes/demo_corpus", key="corpus_index")
    retriever = st.selectbox("Corpus retriever", ["tfidf", "embedding"], key="corpus_retriever")
    if st.button("Build Corpus Index"):
        metadata = build_index_for_corpus(root_path, index_dir, retriever_kind=retriever)
        st.json(metadata.model_dump())

    query = st.text_input(
        "Corpus query",
        "Which models are compared across the notes?",
        key="corpus_query",
    )
    top_k = st.number_input("Corpus Top K", min_value=1, max_value=20, value=5)
    if st.button("Search Corpus"):
        results = search_corpus(index_dir, query, top_k=int(top_k), retriever_kind=retriever)
        st.json(results.model_dump())
    if st.button("Ask Corpus"):
        answer = ask_corpus(index_dir, query, top_k=int(top_k), retriever_kind=retriever)
        st.json(answer.model_dump())
    provider = st.selectbox("Corpus LLM provider", ["fake", "openai"], key="corpus_llm_provider")
    model = st.text_input("Corpus LLM model", "", key="corpus_llm_model")
    trace_path = st.text_input(
        "Corpus trace path",
        "reports/traces/llm_traces.jsonl",
        key="corpus_trace",
    )
    if st.button("LLM Ask Corpus"):
        answer = llm_ask_corpus(
            index_dir,
            query,
            top_k=int(top_k),
            retriever_kind=retriever,
            provider=provider,
            model=model or None,
            trace_path=trace_path or None,
        )
        st.json(answer.model_dump())

with tabs[9]:
    st.subheader("Workflow")
    workflow_index_dir = st.text_input(
        "Workflow index directory",
        "data/indexes/demo_corpus",
        key="workflow_index",
    )
    workflow_query = st.text_input(
        "Workflow query",
        "What experiment is described and can we run a local version?",
        key="workflow_query",
    )
    workflow_retriever = st.selectbox(
        "Workflow retriever",
        ["tfidf", "embedding"],
        key="workflow_retriever",
    )
    workflow_top_k = st.number_input(
        "Workflow Top K",
        min_value=1,
        max_value=20,
        value=5,
        key="workflow_top_k",
    )
    workflow_use_llm = st.checkbox("Use grounded LLM", value=False)
    workflow_provider = st.selectbox("Workflow LLM provider", ["fake", "openai"])
    workflow_model = st.text_input("Workflow LLM model", "")
    workflow_run = st.checkbox("Run bounded experiment if runnable", value=False)
    workflow_seed = st.number_input("Workflow seed", value=42, step=1)
    workflow_out_dir = st.text_input("Workflow output directory", "reports/workflows")
    st.info(
        "This workflow coordinates local retrieval, citation-grounded answering, heuristic "
        "claim/config extraction, validation, and optional bounded runs."
    )
    if st.button("Run Workflow"):
        result = run_research_workflow(
            index_dir=workflow_index_dir,
            query=workflow_query,
            options=WorkflowOptions(
                retriever=workflow_retriever,
                top_k=int(workflow_top_k),
                use_llm=workflow_use_llm,
                llm_provider=workflow_provider,
                llm_model=workflow_model or None,
                run_if_runnable=workflow_run,
                seed=int(workflow_seed),
            ),
            out_dir=workflow_out_dir,
        )
        st.write(result.answer)
        st.json(result.model_dump())
