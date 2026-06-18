from pydantic import BaseModel, Field


class LLMProviderConfig(BaseModel):
    provider: str = "fake"
    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 800


class GroundedLLMAnswer(BaseModel):
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    abstained: bool = False
    reason: str | None = None
    provider: str = "unknown"
    model: str | None = None


class LLMTrace(BaseModel):
    trace_id: str
    provider: str
    model: str | None = None
    query: str
    retrieved_citations: list[str] = Field(default_factory=list)
    used_citations: list[str] = Field(default_factory=list)
    abstained: bool
    latency_ms: float | None = None
    error: str | None = None
