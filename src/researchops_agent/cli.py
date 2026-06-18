import typer

from researchops_agent.agents.extractive import answer_from_evidence
from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document
from researchops_agent.retrieval.evidence import build_evidence_pack
from researchops_agent.retrieval.tfidf import TfidfRetriever

app = typer.Typer(help="ResearchOps Agent CLI")


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
        pages = load_document(path)
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        retriever = TfidfRetriever()
        retriever.fit(chunks)
        retrieval = retriever.search(query, top_k=top_k)
        evidence = build_evidence_pack(retrieval, min_score=min_score)
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


if __name__ == "__main__":
    app()
