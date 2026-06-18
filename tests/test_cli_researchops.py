from typer.testing import CliRunner

from researchops_agent.cli import app
from researchops_agent.utils.json_io import read_json


def test_report_command_works_on_temp_markdown_file(tmp_path) -> None:
    document = tmp_path / "example.md"
    output = tmp_path / "reports" / "report.json"
    document.write_text(
        "We compare Ridge and LSTM on the MIMIC dataset. "
        "Results show Ridge achieves higher accuracy and F1 than the baseline.",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "report",
            str(document),
            "Ridge accuracy F1 MIMIC",
            "--out",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Answer:" in result.output
    assert "Claims:" in result.output
    assert output.exists()
    assert read_json(str(output))["query"] == "Ridge accuracy F1 MIMIC"


def test_claims_command_works_on_temp_markdown_file(tmp_path) -> None:
    document = tmp_path / "example.md"
    document.write_text(
        "We evaluate Ridge on the MIMIC dataset. Results show accuracy improves.",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["claims", str(document), "Ridge accuracy MIMIC"])

    assert result.exit_code == 0
    assert "Claims: 2" in result.output
    assert "Ridge" in result.output
    assert "accuracy" in result.output
