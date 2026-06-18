from researchops_agent.schemas.workflow import ConfigValidationResult, WorkflowOptions


def test_workflow_options_defaults() -> None:
    options = WorkflowOptions()

    assert options.retriever == "tfidf"
    assert options.top_k == 5
    assert options.use_llm is False
    assert options.llm_provider == "fake"
    assert options.run_if_runnable is False
    assert options.seed == 42


def test_config_validation_result_defaults() -> None:
    result = ConfigValidationResult(runnable=False)

    assert result.runnable is False
    assert result.errors == []
