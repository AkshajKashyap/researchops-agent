from pydantic import BaseModel, Field


class RunArtifact(BaseModel):
    name: str
    path: str
    artifact_type: str


class RunMetric(BaseModel):
    name: str
    value: float


class ExperimentRunRequest(BaseModel):
    config_path: str
    out_dir: str = "reports/runs"
    seed: int = 42


class ExperimentRunResult(BaseModel):
    run_id: str
    task: str
    dataset: str | None = None
    models: list[str] = Field(default_factory=list)
    metrics: list[RunMetric] = Field(default_factory=list)
    artifacts: list[RunArtifact] = Field(default_factory=list)
    status: str
    message: str | None = None
