from pydantic import BaseModel, Field


class RetrievalEvalCase(BaseModel):
    case_id: str
    document_path: str
    query: str
    expected_substrings: list[str]
    expected_citations: list[str] = Field(default_factory=list)


class AnswerEvalCase(BaseModel):
    case_id: str
    document_path: str
    query: str
    expected_answer_substrings: list[str] = Field(default_factory=list)
    should_abstain: bool = False


class RetrievalEvalResult(BaseModel):
    case_id: str
    query: str
    top_k: int
    hit: bool
    matched_substrings: list[str] = Field(default_factory=list)
    top_citations: list[str] = Field(default_factory=list)
    top_scores: list[float] = Field(default_factory=list)


class AnswerEvalResult(BaseModel):
    case_id: str
    query: str
    passed: bool
    abstained: bool
    matched_substrings: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    answer: str


class EvaluationSummary(BaseModel):
    retrieval_cases: int
    retrieval_hits: int
    retrieval_hit_rate: float
    answer_cases: int
    answer_passes: int
    answer_pass_rate: float


class EvaluationReport(BaseModel):
    summary: EvaluationSummary
    retrieval_results: list[RetrievalEvalResult] = Field(default_factory=list)
    answer_results: list[AnswerEvalResult] = Field(default_factory=list)
