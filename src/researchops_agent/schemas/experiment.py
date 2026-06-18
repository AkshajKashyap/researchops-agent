from pydantic import BaseModel, Field


class ExperimentClaim(BaseModel):
    task: str | None = None
    dataset: str | None = None
    models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    main_claim: str
    evidence_citations: list[str] = Field(default_factory=list)
    reproducibility_risk: str | None = None


class ExperimentConfig(BaseModel):
    task: str
    dataset: str | None = None
    models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    validation_strategy: str = "not_specified"
    notes: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    claims: list[ExperimentClaim] = Field(default_factory=list)
    suggested_config: ExperimentConfig | None = None
    abstained: bool = False
    reason: str | None = None
