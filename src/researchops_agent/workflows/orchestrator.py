from pathlib import Path

from researchops_agent.agents.claim_extractor import extract_experiment_claims
from researchops_agent.agents.extractive import answer_from_evidence
from researchops_agent.agents.llm_grounded import answer_from_evidence_with_llm
from researchops_agent.corpus.search import search_corpus
from researchops_agent.retrieval.evidence import build_evidence_pack
from researchops_agent.runner.config_builder import suggest_experiment_config
from researchops_agent.runner.experiment_runner import run_experiment_config
from researchops_agent.schemas.llm import LLMProviderConfig
from researchops_agent.schemas.retrieval import RetrievalResult
from researchops_agent.schemas.run import RunArtifact
from researchops_agent.schemas.workflow import WorkflowOptions, WorkflowResult, WorkflowStep
from researchops_agent.utils.json_io import write_json
from researchops_agent.utils.text_io import write_text
from researchops_agent.utils.yaml_io import write_yaml
from researchops_agent.workflows.config_check import check_config_runnable
from researchops_agent.workflows.ids import make_workflow_id
from researchops_agent.workflows.reporting import format_workflow_markdown

HONESTY_NOTES = [
    "The workflow uses only local corpus evidence and does not browse the web.",
    "Experiment claims and suggested configs are produced by deterministic heuristics.",
    "The bounded runner supports only a small explicit set of local toy experiments.",
    "LLM mode is citation-validated, but citations do not guarantee full factual correctness.",
]


def _add_step(steps: list[WorkflowStep], name: str, status: str, message: str | None = None) -> None:
    steps.append(WorkflowStep(name=name, status=status, message=message))


def run_research_workflow(
    index_dir: str,
    query: str,
    options: WorkflowOptions | None = None,
    out_dir: str = "reports/workflows",
) -> WorkflowResult:
    workflow_options = options or WorkflowOptions()
    workflow_id = make_workflow_id(query, index_dir)
    workflow_dir = Path(out_dir) / workflow_id
    steps: list[WorkflowStep] = []

    corpus_result = search_corpus(
        index_dir,
        query,
        top_k=workflow_options.top_k,
        retriever_kind=workflow_options.retriever,
    )
    _add_step(
        steps,
        "corpus_search",
        "success",
        f"Retrieved {len(corpus_result.results)} chunks from corpus {corpus_result.corpus_id}.",
    )

    evidence = build_evidence_pack(
        RetrievalResult(query=corpus_result.query, results=corpus_result.results),
        min_score=0.05,
    )
    _add_step(steps, "evidence_pack", "success", f"Built {len(evidence.items)} evidence items.")

    if workflow_options.use_llm:
        answer = answer_from_evidence_with_llm(
            evidence,
            LLMProviderConfig(
                provider=workflow_options.llm_provider,
                model=workflow_options.llm_model,
            ),
        )
        _add_step(steps, "answering", "success", "Generated citation-validated LLM answer.")
    else:
        answer = answer_from_evidence(evidence)
        _add_step(steps, "answering", "success", "Generated deterministic extractive answer.")

    claims = [] if answer.abstained else extract_experiment_claims(evidence)
    _add_step(steps, "claim_extraction", "success", f"Extracted {len(claims)} experiment claims.")

    suggested_config = None if answer.abstained else suggest_experiment_config(claims)
    if suggested_config is None:
        _add_step(steps, "config_suggestion", "skipped", "No runnable-looking config was suggested.")
    else:
        _add_step(steps, "config_suggestion", "success", "Suggested an experiment config.")

    config_validation = check_config_runnable(suggested_config)
    validation_message = (
        "Suggested config is runnable."
        if config_validation.runnable
        else "; ".join(config_validation.errors)
    )
    _add_step(steps, "config_validation", "success", validation_message)

    run_result = None
    run_artifacts: list[RunArtifact] = []
    if workflow_options.run_if_runnable and suggested_config is not None and config_validation.runnable:
        run_result = run_experiment_config(
            suggested_config,
            out_dir=str(workflow_dir / "runs"),
            seed=workflow_options.seed,
        )
        run_artifacts = run_result.artifacts
        _add_step(steps, "bounded_run", "success", f"Executed bounded run {run_result.run_id}.")
    elif workflow_options.run_if_runnable:
        _add_step(steps, "bounded_run", "skipped", "Suggested config was not runnable.")
    else:
        _add_step(steps, "bounded_run", "skipped", "run_if_runnable was not enabled.")

    workflow_result_path = workflow_dir / "workflow_result.json"
    workflow_summary_path = workflow_dir / "workflow_summary.md"
    artifacts = [
        *run_artifacts,
        RunArtifact(name="workflow_result", path=str(workflow_result_path), artifact_type="json"),
        RunArtifact(name="workflow_summary", path=str(workflow_summary_path), artifact_type="markdown"),
    ]
    suggested_config_path = workflow_dir / "suggested_config.yaml"
    if suggested_config is not None:
        artifacts.append(
            RunArtifact(
                name="suggested_config",
                path=str(suggested_config_path),
                artifact_type="yaml",
            )
        )

    _add_step(steps, "artifact_write", "success", f"Wrote workflow artifacts to {workflow_dir}.")

    result = WorkflowResult(
        workflow_id=workflow_id,
        query=query,
        corpus_index_dir=index_dir,
        options=workflow_options,
        steps=steps,
        answer=answer.answer,
        citations=answer.citations,
        abstained=answer.abstained,
        claims=claims,
        suggested_config=suggested_config,
        config_validation=config_validation,
        run_result=run_result,
        artifacts=artifacts,
        honesty_notes=HONESTY_NOTES,
    )

    if suggested_config is not None:
        write_yaml(str(suggested_config_path), suggested_config.model_dump())
    write_json(str(workflow_result_path), result)
    write_text(str(workflow_summary_path), format_workflow_markdown(result))

    return result
