from researchops_agent.schemas.document import DocumentChunk, DocumentPage


def _chunk_id(path: str, page_number: int | None, chunk_index: int) -> str:
    page_label = page_number if page_number is not None else "none"
    return f"{path}::page:{page_label}::chunk:{chunk_index}"


def chunk_pages(
    pages: list[DocumentPage], chunk_size: int = 1200, overlap: int = 200
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    chunk_counts: dict[tuple[str, int | None], int] = {}
    step = chunk_size - overlap

    for page in pages:
        text = page.text
        if not text.strip():
            continue

        start_char = 0
        while start_char < len(text):
            end_char = min(start_char + chunk_size, len(text))
            chunk_text = text[start_char:end_char]

            if chunk_text.strip():
                chunk_key = (page.source.path, page.page_number)
                chunk_index = chunk_counts.get(chunk_key, 0)
                chunk_counts[chunk_key] = chunk_index + 1

                chunks.append(
                    DocumentChunk(
                        chunk_id=_chunk_id(page.source.path, page.page_number, chunk_index),
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )

            if end_char == len(text):
                break
            start_char += step

    return chunks
