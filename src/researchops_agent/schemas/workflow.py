from pydantic import BaseModel, Field

from researchops_agent.schemas.experiment import ExperimentClaim, ExperimentConfig
from researchops_agent.schemas.run import ExperimentRunResult, RunArtifact


class WorkflowOptions(BaseModel):
    retriever: str = "tfidf"
    top_k: int = 5
    use_llm: bool = False
    llm_provider: str = "fake"
    llm_model: str | None = None
    run_if_runnable: bool = False
    seed: int = 42


class WorkflowStep(BaseModel):
    name: str
    status: str
    message: str | None = None


class ConfigValidationResult(BaseModel):
    runnable: bool
    errors: list[str] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    workflow_id: str
    query: str
    corpus_index_dir: str
    options: WorkflowOptions
    steps: list[WorkflowStep] = Field(default_factory=list)
    answer: str
    citations: list[str] = Field(default_factory=list)
    abstained: bool
    claims: list[ExperimentClaim] = Field(default_factory=list)
    suggested_config: ExperimentConfig | None = None
    config_validation: ConfigValidationResult
    run_result: ExperimentRunResult | None = None
    artifacts: list[RunArtifact] = Field(default_factory=list)
    honesty_notes: list[str] = Field(default_factory=list)
