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
