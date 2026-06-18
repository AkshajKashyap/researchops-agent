from pathlib import Path
from typing import Any

from researchops_agent.utils.json_io import read_json


def list_run_dirs(root: str = "reports/runs") -> list[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(str(path) for path in root_path.iterdir() if path.is_dir())


def load_run_metrics(run_dir: str) -> dict[str, float]:
    metrics_path = Path(run_dir) / "metrics.json"
    data = read_json(str(metrics_path))
    return {metric["name"]: metric["value"] for metric in data.get("metrics", [])}


def summarize_runs(root: str = "reports/runs") -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run_dir in list_run_dirs(root):
        run_path = Path(run_dir)
        try:
            metrics = load_run_metrics(run_dir)
        except FileNotFoundError:
            metrics = {}
        summaries.append(
            {
                "run_id": run_path.name,
                "run_dir": run_dir,
                "metrics": metrics,
            }
        )
    return summaries
