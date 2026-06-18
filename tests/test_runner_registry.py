from researchops_agent.runner.registry import list_run_dirs, summarize_runs
from researchops_agent.utils.json_io import write_json


def test_missing_run_root_returns_empty_list(tmp_path) -> None:
    assert list_run_dirs(str(tmp_path / "missing")) == []
    assert summarize_runs(str(tmp_path / "missing")) == []


def test_summarize_runs_reads_metrics_from_fake_run_directories(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    write_json(
        str(run_dir / "metrics.json"),
        {"metrics": [{"name": "ridge_rmse", "value": 0.12}]},
    )

    summaries = summarize_runs(str(tmp_path / "runs"))

    assert summaries == [
        {
            "run_id": "run-1",
            "run_dir": str(run_dir),
            "metrics": {"ridge_rmse": 0.12},
        }
    ]
