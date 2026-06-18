from researchops_agent.pipeline import (
    ask_document,
    build_report_from_document,
    suggest_config_from_document,
)


def test_ask_document_returns_answer_from_temp_markdown_file(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("Ridge is compared with a baseline using RMSE.", encoding="utf-8")

    answer = ask_document(str(document), "Ridge RMSE")

    assert answer.abstained is False
    assert "Ridge" in answer.answer


def test_build_report_from_document_returns_report(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text("We compare Ridge using RMSE.", encoding="utf-8")

    report = build_report_from_document(str(document), "Ridge RMSE")

    assert report.query == "Ridge RMSE"
    assert report.answer
    assert report.citations


def test_suggest_config_from_document_returns_config_or_none_predictably(tmp_path) -> None:
    document = tmp_path / "note.md"
    document.write_text(
        "We compare mean_baseline and Ridge on the synthetic_sine dataset for "
        "time_series_forecasting using RMSE and MAE with chronological_split validation.",
        encoding="utf-8",
    )

    config = suggest_config_from_document(
        str(document),
        "synthetic_sine time_series_forecasting RMSE MAE",
    )

    assert config is not None
    assert config.task == "time_series_forecasting"
    assert config.dataset == "synthetic_sine"
