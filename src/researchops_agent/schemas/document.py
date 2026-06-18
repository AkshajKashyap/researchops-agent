from typing import Literal

from pydantic import BaseModel


class DocumentSource(BaseModel):
    path: str
    source_type: Literal["pdf", "markdown", "text"]
    title: str | None = None


class DocumentPage(BaseModel):
    source: DocumentSource
    page_number: int | None = None
    text: str


class DocumentChunk(BaseModel):
    chunk_id: str
    source: DocumentSource
    page_number: int | None = None
    chunk_index: int
    text: str
    start_char: int
    end_char: int
