from pydantic import BaseModel


class EvidenceItem(BaseModel):
    chunk_id: str
    citation: str
    score: float
    text: str


class EvidencePack(BaseModel):
    query: str
    items: list[EvidenceItem]


class ExtractiveAnswer(BaseModel):
    query: str
    answer: str
    citations: list[str]
    abstained: bool
    reason: str | None = None
