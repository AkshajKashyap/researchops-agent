from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_path: str
    source_type: str
    page_number: int | None = None
    chunk_index: int
    text: str
    score: float


class RetrievalResult(BaseModel):
    query: str
    results: list[RetrievedChunk]
