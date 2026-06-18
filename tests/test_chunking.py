import pytest

from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.schemas.document import DocumentPage, DocumentSource


def _page(text: str, page_number: int | None = 1) -> DocumentPage:
    source = DocumentSource(path="paper.txt", source_type="text")
    return DocumentPage(source=source, page_number=page_number, text=text)


def test_chunking_creates_multiple_chunks() -> None:
    chunks = chunk_pages([_page("0123456789" * 30)], chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].start_char == 0
    assert chunks[1].start_char == 80


def test_chunk_overlap_works() -> None:
    chunks = chunk_pages([_page("abcdefghijklmnopqrstuvwxyz")], chunk_size=10, overlap=3)

    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[1].start_char == 7


def test_chunk_ids_are_deterministic() -> None:
    pages = [_page("abcdefghijklmnopqrstuvwxyz", page_number=3)]

    first = chunk_pages(pages, chunk_size=10, overlap=3)
    second = chunk_pages(pages, chunk_size=10, overlap=3)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == "paper.txt::page:3::chunk:0"


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_pages([_page("abc")], chunk_size=10, overlap=10)
