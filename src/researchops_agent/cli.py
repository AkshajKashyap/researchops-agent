import typer

from researchops_agent.agents.claim_extractor import extract_experiment_claims
from researchops_agent.agents.extractive import answer_from_evidence
from researchops_agent.agents.report_builder import build_research_report
from researchops_agent.evaluation.answer_eval import evaluate_answers
from researchops_agent.evaluation.load_cases import load_answer_cases, load_retrieval_cases
from researchops_agent.evaluation.report import (
    build_evaluation_report,
    format_evaluation_markdown,
)
from researchops_agent.evaluation.retrieval_eval import evaluate_retrieval
from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document
from researchops_agent.retrieval.evidence import build_evidence_pack
from researchops_agent.retrieval.tfidf import TfidfRetriever
from researchops_agent.schemas.answer import EvidencePack
from researchops_agent.utils.json_io import write_json
from researchops_agent.utils.text_io import write_text

app = typer.Typer(help="ResearchOps Agent CLI")


def _build_evidence_from_document(
    path: str,
    query: str,
    top_k: int,
    min_score: float,
    chunk_size: int,
    overlap: int,
) -> EvidencePack:
    pages = load_document(path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    retriever = TfidfRetriever()
    retriever.fit(chunks)
    retrieval = retriever.search(query, top_k=top_k)
    return build_evidence_pack(retrieval, min_score=min_score)


@app.command()
def health() -> None:
    """Check that the project is wired correctly."""
    typer.echo("researchops-agent is alive")


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
    top_k: int = typer.Option(5, help="Maximum number of retrieval results."),
    chunk_size: int = typer.Option(1200, help="Maximum characters per chunk."),
    overlap: int = typer.Option(200, help="Characters shared between neighboring chunks."),
) -> None:
    """Search a local document with TF-IDF retrieval."""
    try:
        pages = load_document(path)
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        retriever = TfidfRetriever()
        retriever.fit(chunks)
        retrieval = retriever.search(query, top_k=top_k)
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
        retrieval_results = evaluate_retrieval(retrieval_case_data, top_k=top_k)
        answer_results = evaluate_answers(answer_case_data)
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


if __name__ == "__main__":
    app()
