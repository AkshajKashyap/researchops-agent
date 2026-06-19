import importlib.util
from pathlib import Path


DOCS = [
    "docs/architecture.md",
    "docs/safety_and_scope.md",
    "docs/demo_script.md",
    "docs/project_report.md",
    "docs/cli_reference.md",
    "docs/api_reference.md",
    "docs/portfolio_summary.md",
]


def test_docs_files_exist() -> None:
    for path in DOCS:
        assert Path(path).exists()


def test_readme_contains_key_sections() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    for section in [
        "## Quickstart",
        "## Demo Commands",
        "## CLI Overview",
        "## API and Dashboard",
        "## Current Demo Results",
        "## Safety and Scope Limits",
        "## Project Structure",
        "## Future Work",
    ]:
        assert section in readme


def test_demo_artifact_generator_script_exists() -> None:
    assert Path("scripts/generate_demo_artifacts.py").exists()


def test_makefile_contains_final_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "full-check:" in makefile
    assert "demo-all:" in makefile
    assert "ci-check:" in makefile


def test_demo_artifact_generator_runs_with_temp_output(tmp_path) -> None:
    script_path = Path("scripts/generate_demo_artifacts.py")
    spec = importlib.util.spec_from_file_location("generate_demo_artifacts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_dir = tmp_path / "final_demo"

    artifacts = module.generate_demo_artifacts(output_dir=str(output_dir))

    assert Path(artifacts["corpus_eval_json"]).exists()
    assert Path(artifacts["corpus_eval_md"]).exists()
    assert Path(artifacts["evaluation_json"]).exists()
    assert Path(artifacts["evaluation_md"]).exists()
    assert Path(artifacts["run_dir"]).exists()
    assert Path(artifacts["workflow_dir"]).exists()
