from pydantic import BaseModel, Field

from researchops_agent.schemas.retrieval import RetrievedChunk


class CorpusDocument(BaseModel):
    path: str
    source_type: str
    title: str | None = None
    num_pages: int
    num_chunks: int


class CorpusManifest(BaseModel):
    corpus_id: str
    root_path: str
    documents: list[CorpusDocument] = Field(default_factory=list)
    total_pages: int = 0
    total_chunks: int = 0


class CorpusIndexMetadata(BaseModel):
    corpus_id: str
    root_path: str
    retriever: str
    chunk_size: int
    overlap: int
    num_documents: int
    num_chunks: int


class CorpusSearchResult(BaseModel):
    query: str
    corpus_id: str
    results: list[RetrievedChunk] = Field(default_factory=list)
