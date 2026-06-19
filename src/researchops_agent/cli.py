from pathlib import Path

import typer

from researchops_agent import __version__
from researchops_agent.agents.claim_extractor import extract_experiment_claims
from researchops_agent.agents.extractive import answer_from_evidence
from researchops_agent.agents.report_builder import build_research_report
from researchops_agent.evaluation.answer_eval import evaluate_answers
from researchops_agent.evaluation.compare import compare_retrievers
from researchops_agent.evaluation.corpus_eval import evaluate_corpus_retrieval
from researchops_agent.evaluation.llm_answer_eval import evaluate_llm_answers
from researchops_agent.evaluation.load_cases import load_answer_cases, load_retrieval_cases
from researchops_agent.evaluation.report import (
    build_evaluation_report,
    format_evaluation_markdown,
)
from researchops_agent.evaluation.retrieval_eval import evaluate_retrieval
from researchops_agent.pipeline import (
    ask_corpus,
    build_report_from_corpus,
    build_evidence_from_document,
    load_chunk_retrieve,
    llm_ask_corpus,
    llm_ask_document,
    llm_report_from_corpus,
    llm_report_from_document,
    suggest_config_from_document,
)
from researchops_agent.corpus.search import build_index_for_corpus, search_corpus
from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document
from researchops_agent.runner.config_builder import suggest_experiment_config
from researchops_agent.runner.experiment_runner import run_experiment_config
from researchops_agent.schemas.answer import EvidencePack
from researchops_agent.schemas.experiment import ExperimentConfig
from researchops_agent.schemas.workflow import WorkflowOptions
from researchops_agent.utils.json_io import write_json
from researchops_agent.utils.text_io import write_text
from researchops_agent.utils.yaml_io import read_yaml, write_yaml
from researchops_agent.workflows.orchestrator import run_research_workflow

app = typer.Typer(help="ResearchOps Agent CLI")


def _build_evidence_from_document(
    path: str,
    query: str,
    top_k: int,
    min_score: float,
    chunk_size: int,
    overlap: int,
    retriever_kind: str,
) -> EvidencePack:
    if chunk_size == 1200 and overlap == 200:
        return build_evidence_from_document(
            path=path,
            query=query,
            retriever_kind=retriever_kind,
            top_k=top_k,
            min_score=min_score,
        )

    pages = load_document(path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    from researchops_agent.retrieval.evidence import build_evidence_pack
    from researchops_agent.retrieval.factory import build_retriever

    retriever = build_retriever(retriever_kind)
    retriever.fit(chunks)
    return build_evidence_pack(retriever.search(query, top_k=top_k), min_score=min_score)


def _parse_retriever_kinds(retrievers: str) -> list[str]:
    kinds = [kind.strip() for kind in retrievers.split(",") if kind.strip()]
    if not kinds:
        raise ValueError("retrievers must not be empty")
    return kinds


@app.command()
def health() -> None:
    """Check that the project is wired correctly."""
    typer.echo("researchops-agent is alive")


@app.command("version")
def version() -> None:
    """Print the installed researchops-agent version."""
    typer.echo(f"researchops-agent {__version__}")


@app.command()
def ingest(
    path: str,
    chunk_size: int = typer.Option(1200, help="Maximum characters per chunk."),
    overlap: int = typer.Option(200, help="Characters shared between neighboring chunks."),
) -> None:
    """Load and chunk a local document."""
    try:
        pages = load_document(path)
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Pages: {len(pages)}")
    typer.echo(f"Chunks: {len(chunks)}")

    if chunks:
        preview = chunks[0].text[:500].replace("\n", " ")
        typer.echo("First chunk preview:")
        typer.echo(preview)


@app.command()
def retrieve(
    path: str,
    query: str,
    retriever: str = typer.Option("tfidf", help="Retriever backend: tfidf or embedding."),
    top_k: int = typer.Option(5, help="Maximum number of retrieval results."),
    chunk_size: int = typer.Option(1200, help="Maximum characters per chunk."),
    overlap: int = typer.Option(200, help="Characters shared between neighboring chunks."),
) -> None:
    """Search a local document with a selected retrieval backend."""
    try:
        if chunk_size == 1200 and overlap == 200:
            retrieval = load_chunk_retrieve(path, query, retriever_kind=retriever, top_k=top_k)
        else:
            pages = load_document(path)
            chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
            from researchops_agent.retrieval.factory import build_retriever

            retriever_model = build_retriever(retriever)
            retriever_model.fit(chunks)
            retrieval = retriever_model.search(query, top_k=top_k)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    for rank, result in enumerate(retrieval.results, start=1):
        typer.echo(f"Rank: {rank}")
        typer.echo(f"Score: {result.score:.4f}")
        typer.echo(f"Source: {result.source_path}")
        if result.page_number is not None:
            typer.echo(f"Page: {result.page_number}")
        typer.echo(f"Chunk: {result.chunk_index}")
        preview = result.text[:300].replace("\n", " ")
        typer.echo(f"Preview: {preview}")
        if rank < len(retrieval.results):
            typer.echo("")


@app.command()
def ask(
    path: str,
    query: str,
    retriever: str = typer.Option("tfidf", help="Retriever backend: tfidf or embedding."),
    top_k: int = typer.Option(5, help="Maximum number of retrieval results."),
    min_score: float = typer.Option(0.05, help="Minimum retrieval score for evidence."),
    chunk_size: int = typer.Option(1200, help="Maximum characters per chunk."),
    overlap: int = typer.Option(200, help="Characters shared between neighboring chunks."),
) -> None:
    """Answer from retrieved local evidence without generation."""
    try:
        evidence = _build_evidence_from_document(
            path=path,
            query=query,
            top_k=top_k,
            min_score=min_score,
            chunk_size=chunk_size,
            overlap=overlap,
            retriever_kind=retriever,
        )
        answer = answer_from_evidence(evidence)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Answer: {answer.answer}")
    typer.echo(f"Abstained: {answer.abstained}")
    if answer.reason:
        typer.echo(f"Reason: {answer.reason}")
    if answer.citations:
        typer.echo("Citations:")
        for citation in answer.citations:
            typer.echo(f"- {citation}")


@app.command()
def claims(
    path: str,
    query: str,
    retriever: str = typer.Option("tfidf", help="Retriever backend: tfidf or embedding."),
    top_k: int = typer.Option(5, help="Maximum number of retrieval results."),
    min_score: float = typer.Option(0.05, help="Minimum retrieval score for evidence."),
    chunk_size: int = typer.Option(1200, help="Maximum characters per chunk."),
    overlap: int = typer.Option(200, help="Characters shared between neighboring chunks."),
) -> None:
    """Extract experiment-like claims from local retrieved evidence."""
    try:
        evidence = _build_evidence_from_document(
            path=path,
            query=query,
            top_k=top_k,
            min_score=min_score,
            chunk_size=chunk_size,
            overlap=overlap,
            retriever_kind=retriever,
        )
        extracted_claims = extract_experiment_claims(evidence)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Claims: {len(extracted_claims)}")
    for index, claim in enumerate(extracted_claims, start=1):
        typer.echo(f"Claim {index}: {claim.main_claim}")
        typer.echo(f"Models: {', '.join(claim.models) if claim.models else 'not_specified'}")
        typer.echo(f"Metrics: {', '.join(claim.metrics) if claim.metrics else 'not_specified'}")
        typer.echo(f"Dataset: {claim.dataset or 'not_specified'}")
        typer.echo("Citations:")
        for citation in claim.evidence_citations:
            typer.echo(f"- {citation}")
        if claim.reproducibility_risk:
            typer.echo(f"Reproducibility risk: {claim.reproducibility_risk}")
        if index < len(extracted_claims):
            typer.echo("")


@app.command()
def report(
    path: str,
    query: str,
    out: str | None = typer.Option(None, "--out", help="Write the full report to JSON."),
    retriever: str = typer.Option("tfidf", help="Retriever backend: tfidf or embedding."),
    top_k: int = typer.Option(5, help="Maximum number of retrieval results."),
    min_score: float = typer.Option(0.05, help="Minimum retrieval score for evidence."),
    chunk_size: int = typer.Option(1200, help="Maximum characters per chunk."),
    overlap: int = typer.Option(200, help="Characters shared between neighboring chunks."),
) -> None:
    """Build a local JSON-ready research report from retrieved evidence."""
    try:
        evidence = _build_evidence_from_document(
            path=path,
            query=query,
            top_k=top_k,
            min_score=min_score,
            chunk_size=chunk_size,
            overlap=overlap,
            retriever_kind=retriever,
        )
        answer = answer_from_evidence(evidence)
        extracted_claims = extract_experiment_claims(evidence)
        research_report = build_research_report(query, answer, extracted_claims)
        if out:
            write_json(out, research_report)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Answer: {research_report.answer}")
    if research_report.citations:
        typer.echo("Citations:")
        for citation in research_report.citations:
            typer.echo(f"- {citation}")
    typer.echo(f"Claims: {len(research_report.claims)}")

    if research_report.suggested_config:
        config = research_report.suggested_config
        typer.echo("Suggested config:")
        typer.echo(f"Task: {config.task}")
        typer.echo(f"Dataset: {config.dataset or 'not_specified'}")
        typer.echo(f"Models: {', '.join(config.models) if config.models else 'not_specified'}")
        typer.echo(f"Metrics: {', '.join(config.metrics) if config.metrics else 'not_specified'}")
        typer.echo(f"Validation: {config.validation_strategy}")
    else:
        typer.echo("Suggested config: none")

    if research_report.abstained:
        typer.echo("Abstained: True")
        if research_report.reason:
            typer.echo(f"Reason: {research_report.reason}")
    if out:
        typer.echo(f"Wrote report: {out}")


@app.command("eval")
def eval_command(
    retriever: str = typer.Option(
        "tfidf",
        "--retriever",
        help="Retriever backend: tfidf or embedding.",
    ),
    retrieval_cases: str = typer.Option(
        "examples/eval/retrieval_cases.json",
        "--retrieval-cases",
        help="Path to retrieval evaluation cases JSON.",
    ),
    answer_cases: str = typer.Option(
        "examples/eval/answer_cases.json",
        "--answer-cases",
        help="Path to answer evaluation cases JSON.",
    ),
    out_json: str | None = typer.Option(
        "reports/evaluation.json",
        "--out-json",
        help="Write the evaluation report to JSON.",
    ),
    out_md: str | None = typer.Option(
        "reports/evaluation.md",
        "--out-md",
        help="Write the evaluation report to Markdown.",
    ),
    top_k: int = typer.Option(3, "--top-k", help="Number of retrieval results to evaluate."),
) -> None:
    """Run local deterministic retrieval and answer evaluations."""
    try:
        retrieval_case_data = load_retrieval_cases(retrieval_cases)
        answer_case_data = load_answer_cases(answer_cases)
        retrieval_results = evaluate_retrieval(
            retrieval_case_data, top_k=top_k, retriever_kind=retriever
        )
        answer_results = evaluate_answers(answer_case_data, retriever_kind=retriever)
        evaluation_report = build_evaluation_report(retrieval_results, answer_results)

        if out_json:
            write_json(out_json, evaluation_report)
        if out_md:
            write_text(out_md, format_evaluation_markdown(evaluation_report))
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = evaluation_report.summary
    typer.echo(
        f"Retrieval hit rate: {summary.retrieval_hit_rate:.2f} "
        f"({summary.retrieval_hits}/{summary.retrieval_cases})"
    )
    typer.echo(
        f"Answer pass rate: {summary.answer_pass_rate:.2f} "
        f"({summary.answer_passes}/{summary.answer_cases})"
    )
    typer.echo(f"Cases: {summary.retrieval_cases} retrieval, {summary.answer_cases} answer")
    if out_json:
        typer.echo(f"Wrote JSON report: {out_json}")
    if out_md:
        typer.echo(f"Wrote Markdown report: {out_md}")


@app.command("compare-retrievers")
def compare_retrievers_command(
    retrieval_cases: str = typer.Option(
        "examples/eval/retrieval_cases.json",
        "--retrieval-cases",
        help="Path to retrieval evaluation cases JSON.",
    ),
    answer_cases: str = typer.Option(
        "examples/eval/answer_cases.json",
        "--answer-cases",
        help="Path to answer evaluation cases JSON.",
    ),
    retrievers: str = typer.Option(
        "tfidf,embedding",
        "--retrievers",
        help="Comma-separated retriever backends to compare.",
    ),
    top_k: int = typer.Option(3, "--top-k", help="Number of retrieval results to evaluate."),
    out_dir: str = typer.Option(
        "reports/retriever_comparison",
        "--out-dir",
        help="Directory for per-retriever evaluation reports.",
    ),
) -> None:
    """Compare local retriever backends on the evaluation cases."""
    try:
        retriever_kinds = _parse_retriever_kinds(retrievers)
        retrieval_case_data = load_retrieval_cases(retrieval_cases)
        answer_case_data = load_answer_cases(answer_cases)
        reports = compare_retrievers(
            retrieval_cases=retrieval_case_data,
            answer_cases=answer_case_data,
            retriever_kinds=retriever_kinds,
            top_k=top_k,
        )

        output_dir = Path(out_dir)
        for retriever_kind, evaluation_report in reports.items():
            json_path = output_dir / f"{retriever_kind}_evaluation.json"
            markdown_path = output_dir / f"{retriever_kind}_evaluation.md"
            write_json(str(json_path), evaluation_report)
            write_text(str(markdown_path), format_evaluation_markdown(evaluation_report))
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo("| Retriever | Retrieval hit rate | Answer pass rate | Retrieval cases | Answer cases |")
    typer.echo("| --- | ---: | ---: | ---: | ---: |")
    for retriever_kind, evaluation_report in reports.items():
        summary = evaluation_report.summary
        typer.echo(
            f"| {retriever_kind} | "
            f"{summary.retrieval_hit_rate:.2f} | "
            f"{summary.answer_pass_rate:.2f} | "
            f"{summary.retrieval_cases} | "
            f"{summary.answer_cases} |"
        )
    typer.echo(f"Wrote comparison reports: {out_dir}")


@app.command("run-config")
def run_config_command(
    config_path: str,
    out_dir: str = typer.Option("reports/runs", "--out-dir", help="Directory for run outputs."),
    seed: int = typer.Option(42, "--seed", help="Random seed for deterministic data generation."),
) -> None:
    """Run a supported bounded local experiment config."""
    try:
        config = ExperimentConfig.model_validate(read_yaml(config_path))
        result = run_experiment_config(config, out_dir=out_dir, seed=seed)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Status: {result.status}")
    typer.echo("Metrics:")
    for metric in result.metrics:
        typer.echo(f"- {metric.name}: {metric.value:.6f}")
    typer.echo("Artifacts:")
    for artifact in result.artifacts:
        typer.echo(f"- {artifact.name} ({artifact.artifact_type}): {artifact.path}")


@app.command("suggest-config")
def suggest_config_command(
    path: str,
    query: str,
    out: str | None = typer.Option(None, "--out", help="Write suggested config to YAML."),
    retriever: str = typer.Option("tfidf", help="Retriever backend: tfidf or embedding."),
    top_k: int = typer.Option(5, help="Maximum number of retrieval results."),
    min_score: float = typer.Option(0.05, help="Minimum retrieval score for evidence."),
    chunk_size: int = typer.Option(1200, help="Maximum characters per chunk."),
    overlap: int = typer.Option(200, help="Characters shared between neighboring chunks."),
) -> None:
    """Suggest a heuristic experiment config from local document evidence."""
    try:
        if top_k == 5 and min_score == 0.05 and chunk_size == 1200 and overlap == 200:
            config = suggest_config_from_document(path, query, retriever_kind=retriever)
        else:
            evidence = _build_evidence_from_document(
                path=path,
                query=query,
                top_k=top_k,
                min_score=min_score,
                chunk_size=chunk_size,
                overlap=overlap,
                retriever_kind=retriever,
            )
            extracted_claims = extract_experiment_claims(evidence)
            config = suggest_experiment_config(extracted_claims)
        if config is not None and out:
            write_yaml(out, config.model_dump())
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if config is None:
        typer.echo("No experiment config could be suggested from retrieved evidence.")
        return

    typer.echo("Suggested config:")
    typer.echo(f"task: {config.task}")
    typer.echo(f"dataset: {config.dataset or 'not_specified'}")
    typer.echo(f"models: {', '.join(config.models) if config.models else 'not_specified'}")
    typer.echo(f"metrics: {', '.join(config.metrics) if config.metrics else 'not_specified'}")
    typer.echo(f"validation_strategy: {config.validation_strategy}")
    if out:
        typer.echo(f"Wrote config: {out}")


@app.command("llm-ask")
def llm_ask_command(
    path: str,
    query: str,
    retriever: str = typer.Option("tfidf", "--retriever", help="Retriever backend."),
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval results."),
    provider: str = typer.Option("fake", "--provider", help="LLM provider: fake or openai."),
    model: str | None = typer.Option(None, "--model", help="LLM model name."),
    trace: str | None = typer.Option(
        "reports/traces/llm_traces.jsonl",
        "--trace",
        help="Path for JSONL trace metadata.",
    ),
) -> None:
    """Answer with an optional grounded LLM provider."""
    try:
        answer = llm_ask_document(
            path,
            query,
            retriever_kind=retriever,
            top_k=top_k,
            provider=provider,
            model=model,
            trace_path=trace,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Answer: {answer.answer}")
    typer.echo(f"Abstained: {answer.abstained}")
    if answer.reason:
        typer.echo(f"Reason: {answer.reason}")
    if answer.citations:
        typer.echo("Citations:")
        for citation in answer.citations:
            typer.echo(f"- {citation}")


@app.command("llm-report")
def llm_report_command(
    path: str,
    query: str,
    retriever: str = typer.Option("tfidf", "--retriever", help="Retriever backend."),
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval results."),
    provider: str = typer.Option("fake", "--provider", help="LLM provider: fake or openai."),
    model: str | None = typer.Option(None, "--model", help="LLM model name."),
    out: str | None = typer.Option(
        "reports/llm_report.json",
        "--out",
        help="Write the LLM report to JSON.",
    ),
    trace: str | None = typer.Option(
        "reports/traces/llm_traces.jsonl",
        "--trace",
        help="Path for JSONL trace metadata.",
    ),
) -> None:
    """Build a report with an optional grounded LLM answer."""
    try:
        report = llm_report_from_document(
            path,
            query,
            retriever_kind=retriever,
            top_k=top_k,
            provider=provider,
            model=model,
            trace_path=trace,
        )
        if out:
            write_json(out, report)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Answer: {report.answer}")
    if report.citations:
        typer.echo("Citations:")
        for citation in report.citations:
            typer.echo(f"- {citation}")
    typer.echo(f"Claims: {len(report.claims)}")
    if report.suggested_config:
        config = report.suggested_config
        typer.echo("Suggested config:")
        typer.echo(f"Task: {config.task}")
        typer.echo(f"Dataset: {config.dataset or 'not_specified'}")
        typer.echo(f"Models: {', '.join(config.models) if config.models else 'not_specified'}")
        typer.echo(f"Metrics: {', '.join(config.metrics) if config.metrics else 'not_specified'}")
        typer.echo(f"Validation: {config.validation_strategy}")
    else:
        typer.echo("Suggested config: none")
    if out:
        typer.echo(f"Wrote report: {out}")


@app.command("eval-llm")
def eval_llm_command(
    answer_cases: str = typer.Option(
        "examples/eval/answer_cases.json",
        "--answer-cases",
        help="Path to answer evaluation cases JSON.",
    ),
    retriever: str = typer.Option("tfidf", "--retriever", help="Retriever backend."),
    provider: str = typer.Option("fake", "--provider", help="LLM provider: fake or openai."),
    model: str | None = typer.Option(None, "--model", help="LLM model name."),
    out_json: str | None = typer.Option(
        "reports/llm_answer_eval.json",
        "--out-json",
        help="Write LLM answer evaluation to JSON.",
    ),
    out_md: str | None = typer.Option(
        "reports/llm_answer_eval.md",
        "--out-md",
        help="Write LLM answer evaluation to Markdown.",
    ),
) -> None:
    """Evaluate grounded LLM answers against local answer cases."""
    try:
        answer_case_data = load_answer_cases(answer_cases)
        answer_results = evaluate_llm_answers(
            answer_case_data,
            retriever_kind=retriever,
            provider=provider,
            model=model,
        )
        evaluation_report = build_evaluation_report([], answer_results)
        if out_json:
            write_json(out_json, evaluation_report)
        if out_md:
            write_text(out_md, format_evaluation_markdown(evaluation_report))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = evaluation_report.summary
    typer.echo(
        f"LLM answer pass rate: {summary.answer_pass_rate:.2f} "
        f"({summary.answer_passes}/{summary.answer_cases})"
    )
    if out_json:
        typer.echo(f"Wrote JSON report: {out_json}")
    if out_md:
        typer.echo(f"Wrote Markdown report: {out_md}")


@app.command("index-corpus")
def index_corpus_command(
    root_path: str,
    index_dir: str = typer.Option(
        "data/indexes/demo_corpus",
        "--index-dir",
        help="Directory for the persisted corpus index.",
    ),
    retriever: str = typer.Option("tfidf", "--retriever", help="Retriever backend."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Search folders recursively."),
    chunk_size: int = typer.Option(1200, "--chunk-size", help="Maximum characters per chunk."),
    overlap: int = typer.Option(200, "--overlap", help="Characters shared between chunks."),
) -> None:
    """Build a persistent local corpus index from documents."""
    try:
        metadata = build_index_for_corpus(
            root_path=root_path,
            index_dir=index_dir,
            retriever_kind=retriever,
            recursive=recursive,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Corpus ID: {metadata.corpus_id}")
    typer.echo(f"Documents: {metadata.num_documents}")
    typer.echo(f"Chunks: {metadata.num_chunks}")
    typer.echo(f"Retriever: {metadata.retriever}")
    typer.echo(f"Index directory: {index_dir}")


@app.command("search-corpus")
def search_corpus_command(
    index_dir: str,
    query: str,
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval results."),
    retriever: str | None = typer.Option(None, "--retriever", help="Override retriever backend."),
) -> None:
    """Search a persisted local corpus index."""
    try:
        result = search_corpus(index_dir, query, top_k=top_k, retriever_kind=retriever)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    for rank, item in enumerate(result.results, start=1):
        typer.echo(f"Rank: {rank}")
        typer.echo(f"Score: {item.score:.4f}")
        typer.echo(f"Source: {item.source_path}")
        if item.page_number is not None:
            typer.echo(f"Page: {item.page_number}")
        typer.echo(f"Chunk: {item.chunk_index}")
        typer.echo(f"Preview: {item.text[:300].replace(chr(10), ' ')}")
        if rank < len(result.results):
            typer.echo("")


@app.command("ask-corpus")
def ask_corpus_command(
    index_dir: str,
    query: str,
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval results."),
    retriever: str | None = typer.Option(None, "--retriever", help="Override retriever backend."),
) -> None:
    """Answer from persisted corpus evidence with the extractive baseline."""
    try:
        answer = ask_corpus(index_dir, query, top_k=top_k, retriever_kind=retriever)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Answer: {answer.answer}")
    typer.echo(f"Abstained: {answer.abstained}")
    if answer.reason:
        typer.echo(f"Reason: {answer.reason}")
    if answer.citations:
        typer.echo("Citations:")
        for citation in answer.citations:
            typer.echo(f"- {citation}")


@app.command("llm-ask-corpus")
def llm_ask_corpus_command(
    index_dir: str,
    query: str,
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval results."),
    retriever: str | None = typer.Option(None, "--retriever", help="Override retriever backend."),
    provider: str = typer.Option("fake", "--provider", help="LLM provider: fake or openai."),
    model: str | None = typer.Option(None, "--model", help="LLM model name."),
    trace: str | None = typer.Option(
        "reports/traces/llm_traces.jsonl",
        "--trace",
        help="Path for JSONL trace metadata.",
    ),
) -> None:
    """Answer from corpus evidence with an optional grounded LLM provider."""
    try:
        answer = llm_ask_corpus(
            index_dir,
            query,
            top_k=top_k,
            retriever_kind=retriever,
            provider=provider,
            model=model,
            trace_path=trace,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Answer: {answer.answer}")
    typer.echo(f"Abstained: {answer.abstained}")
    if answer.reason:
        typer.echo(f"Reason: {answer.reason}")
    if answer.citations:
        typer.echo("Citations:")
        for citation in answer.citations:
            typer.echo(f"- {citation}")


@app.command("corpus-report")
def corpus_report_command(
    index_dir: str,
    query: str,
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval results."),
    retriever: str | None = typer.Option(None, "--retriever", help="Override retriever backend."),
    out: str | None = typer.Option(
        "reports/corpus_report.json",
        "--out",
        help="Write corpus report to JSON.",
    ),
) -> None:
    """Build a deterministic corpus report."""
    try:
        report = build_report_from_corpus(index_dir, query, top_k=top_k, retriever_kind=retriever)
        if out:
            write_json(out, report)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Answer: {report.answer}")
    typer.echo(f"Claims: {len(report.claims)}")
    if out:
        typer.echo(f"Wrote report: {out}")


@app.command("llm-corpus-report")
def llm_corpus_report_command(
    index_dir: str,
    query: str,
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval results."),
    retriever: str | None = typer.Option(None, "--retriever", help="Override retriever backend."),
    provider: str = typer.Option("fake", "--provider", help="LLM provider: fake or openai."),
    model: str | None = typer.Option(None, "--model", help="LLM model name."),
    out: str | None = typer.Option(
        "reports/llm_corpus_report.json",
        "--out",
        help="Write LLM corpus report to JSON.",
    ),
    trace: str | None = typer.Option(
        "reports/traces/llm_traces.jsonl",
        "--trace",
        help="Path for JSONL trace metadata.",
    ),
) -> None:
    """Build a grounded LLM corpus report."""
    try:
        report = llm_report_from_corpus(
            index_dir,
            query,
            top_k=top_k,
            retriever_kind=retriever,
            provider=provider,
            model=model,
            trace_path=trace,
        )
        if out:
            write_json(out, report)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Answer: {report.answer}")
    typer.echo(f"Claims: {len(report.claims)}")
    if out:
        typer.echo(f"Wrote report: {out}")


@app.command("eval-corpus")
def eval_corpus_command(
    index_dir: str = typer.Option(
        "data/indexes/demo_corpus",
        "--index-dir",
        help="Path to persisted corpus index.",
    ),
    cases: str = typer.Option(
        "examples/eval/corpus_retrieval_cases.json",
        "--cases",
        help="Path to corpus retrieval cases JSON.",
    ),
    out_json: str | None = typer.Option(
        "reports/corpus_eval.json",
        "--out-json",
        help="Write corpus eval report to JSON.",
    ),
    out_md: str | None = typer.Option(
        "reports/corpus_eval.md",
        "--out-md",
        help="Write corpus eval report to Markdown.",
    ),
    top_k: int = typer.Option(3, "--top-k", help="Maximum number of retrieval results."),
    retriever: str | None = typer.Option(None, "--retriever", help="Override retriever backend."),
) -> None:
    """Evaluate retrieval over a persisted corpus index."""
    try:
        retrieval_cases = load_retrieval_cases(cases)
        retrieval_results = evaluate_corpus_retrieval(
            index_dir,
            retrieval_cases,
            top_k=top_k,
            retriever_kind=retriever,
        )
        evaluation_report = build_evaluation_report(retrieval_results, [])
        if out_json:
            write_json(out_json, evaluation_report)
        if out_md:
            write_text(out_md, format_evaluation_markdown(evaluation_report))
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = evaluation_report.summary
    typer.echo(
        f"Corpus retrieval hit rate: {summary.retrieval_hit_rate:.2f} "
        f"({summary.retrieval_hits}/{summary.retrieval_cases})"
    )
    if out_json:
        typer.echo(f"Wrote JSON report: {out_json}")
    if out_md:
        typer.echo(f"Wrote Markdown report: {out_md}")


@app.command("workflow")
def workflow_command(
    index_dir: str,
    query: str,
    retriever: str = typer.Option("tfidf", "--retriever", help="Retriever backend."),
    top_k: int = typer.Option(5, "--top-k", help="Maximum number of retrieval results."),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use grounded LLM answering."),
    llm_provider: str = typer.Option("fake", "--llm-provider", help="LLM provider: fake or openai."),
    llm_model: str | None = typer.Option(None, "--llm-model", help="LLM model name."),
    run_if_runnable: bool = typer.Option(
        False,
        "--run-if-runnable",
        help="Run the bounded experiment if the suggested config validates.",
    ),
    seed: int = typer.Option(42, "--seed", help="Seed for bounded experiment runs."),
    out_dir: str = typer.Option(
        "reports/workflows",
        "--out-dir",
        help="Directory for workflow artifacts.",
    ),
) -> None:
    """Run the end-to-end local ResearchOps workflow over a corpus index."""
    try:
        result = run_research_workflow(
            index_dir=index_dir,
            query=query,
            options=WorkflowOptions(
                retriever=retriever,
                top_k=top_k,
                use_llm=use_llm,
                llm_provider=llm_provider,
                llm_model=llm_model,
                run_if_runnable=run_if_runnable,
                seed=seed,
            ),
            out_dir=out_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Workflow ID: {result.workflow_id}")
    typer.echo(f"Answer: {result.answer}")
    typer.echo(f"Abstained: {result.abstained}")
    if result.citations:
        typer.echo("Citations:")
        for citation in result.citations:
            typer.echo(f"- {citation}")
    typer.echo(f"Claims: {len(result.claims)}")
    typer.echo(f"Config runnable: {result.config_validation.runnable}")
    if result.config_validation.errors:
        typer.echo("Config validation errors:")
        for error in result.config_validation.errors:
            typer.echo(f"- {error}")
    typer.echo(f"Run executed: {result.run_result is not None}")
    typer.echo("Artifacts:")
    for artifact in result.artifacts:
        typer.echo(f"- {artifact.name} ({artifact.artifact_type}): {artifact.path}")


if __name__ == "__main__":
    app()
