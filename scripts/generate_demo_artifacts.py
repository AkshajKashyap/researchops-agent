from __future__ import annotations

import argparse
from pathlib import Path

from researchops_agent.corpus.search import build_index_for_corpus
from researchops_agent.evaluation.answer_eval import evaluate_answers
from researchops_agent.evaluation.corpus_eval import evaluate_corpus_retrieval
from researchops_agent.evaluation.load_cases import load_answer_cases, load_retrieval_cases
from researchops_agent.evaluation.report import build_evaluation_report, format_evaluation_markdown
from researchops_agent.evaluation.retrieval_eval import evaluate_retrieval
from researchops_agent.runner.experiment_runner import run_experiment_config
from researchops_agent.schemas.experiment import ExperimentConfig
from researchops_agent.schemas.workflow import WorkflowOptions
from researchops_agent.utils.json_io import write_json
from researchops_agent.utils.text_io import write_text
from researchops_agent.utils.yaml_io import read_yaml
from researchops_agent.workflows.orchestrator import run_research_workflow


def generate_demo_artifacts(
    output_dir: str = "reports/final_demo",
    index_dir: str | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index_path = Path(index_dir) if index_dir else output_path / "indexes" / "demo_corpus"

    metadata = build_index_for_corpus(
        root_path="examples/docs",
        index_dir=str(index_path),
        retriever_kind="tfidf",
    )

    corpus_cases = load_retrieval_cases("examples/eval/corpus_retrieval_cases.json")
    corpus_results = evaluate_corpus_retrieval(str(index_path), corpus_cases, top_k=3)
    corpus_report = build_evaluation_report(corpus_results, [])
    corpus_json = output_path / "corpus_eval.json"
    corpus_md = output_path / "corpus_eval.md"
    write_json(str(corpus_json), corpus_report)
    write_text(str(corpus_md), format_evaluation_markdown(corpus_report))

    retrieval_cases = load_retrieval_cases("examples/eval/retrieval_cases.json")
    answer_cases = load_answer_cases("examples/eval/answer_cases.json")
    retrieval_results = evaluate_retrieval(retrieval_cases, top_k=3, retriever_kind="tfidf")
    answer_results = evaluate_answers(answer_cases, retriever_kind="tfidf")
    evaluation_report = build_evaluation_report(retrieval_results, answer_results)
    evaluation_json = output_path / "evaluation.json"
    evaluation_md = output_path / "evaluation.md"
    write_json(str(evaluation_json), evaluation_report)
    write_text(str(evaluation_md), format_evaluation_markdown(evaluation_report))

    config = ExperimentConfig.model_validate(read_yaml("configs/time_series_demo.yaml"))
    run_result = run_experiment_config(config, out_dir=str(output_path / "runs"), seed=42)

    workflow_result = run_research_workflow(
        index_dir=str(index_path),
        query="What experiment is described and can we run a local version?",
        options=WorkflowOptions(
            retriever="tfidf",
            top_k=5,
            use_llm=True,
            llm_provider="fake",
            run_if_runnable=True,
            seed=42,
        ),
        out_dir=str(output_path / "workflows"),
    )

    return {
        "index_dir": str(index_path),
        "corpus_id": metadata.corpus_id,
        "corpus_eval_json": str(corpus_json),
        "corpus_eval_md": str(corpus_md),
        "evaluation_json": str(evaluation_json),
        "evaluation_md": str(evaluation_md),
        "run_id": run_result.run_id,
        "run_dir": str(output_path / "runs" / run_result.run_id),
        "workflow_id": workflow_result.workflow_id,
        "workflow_dir": str(output_path / "workflows" / workflow_result.workflow_id),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local ResearchOps demo artifacts.")
    parser.add_argument("--out-dir", default="reports/final_demo", help="Output directory.")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Optional corpus index directory. Defaults to OUT_DIR/indexes/demo_corpus.",
    )
    args = parser.parse_args()

    artifacts = generate_demo_artifacts(output_dir=args.out_dir, index_dir=args.index_dir)
    print("Generated final demo artifacts:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
