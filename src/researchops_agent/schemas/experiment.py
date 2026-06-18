from pydantic import BaseModel, Field


class ExperimentClaim(BaseModel):
    task: str
    dataset: str | None = None
    models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    main_claim: str
    reproducibility_risk: str | None = None
