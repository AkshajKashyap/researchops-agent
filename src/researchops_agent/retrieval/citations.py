from researchops_agent.schemas.retrieval import RetrievedChunk


def format_citation(result: RetrievedChunk) -> str:
    if result.page_number is not None:
        return f"{result.source_path}#page={result.page_number}&chunk={result.chunk_index}"
    return f"{result.source_path}#chunk={result.chunk_index}"
