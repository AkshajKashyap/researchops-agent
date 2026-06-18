from pydantic import BaseModel


class DocumentQueryRequest(BaseModel):
    path: str
    query: str
    retriever: str = "tfidf"
    top_k: int = 5


class RunConfigRequest(BaseModel):
    config_path: str
    out_dir: str = "reports/runs"
    seed: int = 42


class EvalRequest(BaseModel):
    retrieval_cases: str = "examples/eval/retrieval_cases.json"
    answer_cases: str = "examples/eval/answer_cases.json"
    retriever: str = "tfidf"
    top_k: int = 3


class LLMDocumentQueryRequest(BaseModel):
    path: str
    query: str
    retriever: str = "tfidf"
    top_k: int = 5
    provider: str = "fake"
    model: str | None = None
    trace_path: str | None = None


class CorpusIndexRequest(BaseModel):
    root_path: str
    index_dir: str = "data/indexes/demo_corpus"
    retriever: str = "tfidf"
    recursive: bool = True
    chunk_size: int = 1200
    overlap: int = 200


class CorpusQueryRequest(BaseModel):
    index_dir: str
    query: str
    top_k: int = 5
    retriever: str | None = None


class LLMCorpusQueryRequest(BaseModel):
    index_dir: str
    query: str
    top_k: int = 5
    retriever: str | None = None
    provider: str = "fake"
    model: str | None = None
    trace_path: str | None = None
