import pytest
from fastapi import HTTPException

from researchops_agent.api.main import corpus_index, workflow
from researchops_agent.schemas.api import CorpusIndexRequest, WorkflowRequest


def test_api_workflow_works_on_temp_corpus_index(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text(
        "We compare mean_baseline and Ridge for time_series_forecasting on synthetic_sine "
        "dataset using RMSE and MAE with chronological_split validation.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    corpus_index(CorpusIndexRequest(root_path=str(docs), index_dir=str(index_dir)))

    response = workflow(
        WorkflowRequest(
            index_dir=str(index_dir),
            query="time_series_forecasting synthetic_sine RMSE",
            out_dir=str(tmp_path / "workflows"),
        )
    )

    assert response.answer
    assert response.citations
    assert response.artifacts


def test_api_workflow_invalid_index_returns_http_400(tmp_path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        workflow(
            WorkflowRequest(
                index_dir=str(tmp_path / "missing"),
                query="What happened?",
                out_dir=str(tmp_path / "workflows"),
            )
        )

    assert exc_info.value.status_code == 400
