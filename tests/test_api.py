import pytest
from fastapi import HTTPException

from researchops_agent.api.main import ask, health, llm_ask, llm_report, retrieve, run_config
from researchops_agent.schemas.api import DocumentQueryRequest, LLMDocumentQueryRequest, RunConfigRequest
from researchops_agent.utils.yaml_io import write_yaml


def test_api_health_works() -> None:
    assert health() == {"status": "ok"}


def test_api_ask_works_on_temp_markdown_file(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge is compared using RMSE.", encoding="utf-8")

    response = ask(
        DocumentQueryRequest(path=str(document), query="Ridge RMSE")
    )

    assert response.abstained is False


def test_api_retrieve_works_on_temp_markdown_file(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge is compared using RMSE.", encoding="utf-8")

    response = retrieve(
        DocumentQueryRequest(path=str(document), query="Ridge RMSE")
    )

    assert response.results[0].source_path == str(document)


def test_api_run_config_works_on_temp_yaml_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "runs"
    write_yaml(
        str(config_path),
        {
            "task": "time_series_forecasting",
            "dataset": "synthetic_sine",
            "models": ["mean_baseline", "ridge"],
            "metrics": ["rmse", "mae"],
            "validation_strategy": "chronological_split",
        },
    )

    response = run_config(
        RunConfigRequest(config_path=str(config_path), out_dir=str(out_dir), seed=123)
    )

    assert response.status == "success"
    assert response.metrics


def test_api_invalid_retriever_returns_http_400(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge is compared using RMSE.", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        ask(DocumentQueryRequest(path=str(document), query="Ridge RMSE", retriever="unknown"))

    assert exc_info.value.status_code == 400


def test_api_invalid_config_returns_http_400(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    write_yaml(
        str(config_path),
        {
            "task": "classification",
            "dataset": "synthetic_sine",
            "models": ["ridge"],
            "metrics": ["rmse"],
            "validation_strategy": "chronological_split",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        run_config(RunConfigRequest(config_path=str(config_path)))

    assert exc_info.value.status_code == 400


def test_api_llm_ask_works_with_fake_provider(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge and LSTM are compared using RMSE.", encoding="utf-8")

    response = llm_ask(
        LLMDocumentQueryRequest(
            path=str(document),
            query="What models were compared?",
            provider="fake",
        )
    )

    assert response.abstained is False
    assert response.citations


def test_api_llm_report_works_with_fake_provider(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("We compare Ridge using RMSE.", encoding="utf-8")

    response = llm_report(
        LLMDocumentQueryRequest(
            path=str(document),
            query="Ridge RMSE",
            provider="fake",
        )
    )

    assert response.answer
    assert response.citations


def test_api_llm_invalid_provider_returns_http_400(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge is compared using RMSE.", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        llm_ask(
            LLMDocumentQueryRequest(
                path=str(document),
                query="Ridge RMSE",
                provider="unknown",
            )
        )

    assert exc_info.value.status_code == 400
