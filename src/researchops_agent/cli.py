import typer

from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document

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


if __name__ == "__main__":
    app()
