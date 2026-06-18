import typer

app = typer.Typer(help="ResearchOps Agent CLI")

@app.command()
def health() -> None:
"""Check that the project is wired correctly."""
typer.echo("researchops-agent is alive")

if name == "main":
app()
