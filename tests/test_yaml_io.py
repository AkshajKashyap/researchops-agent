from researchops_agent.utils.yaml_io import read_yaml, write_yaml


def test_write_yaml_creates_parent_dirs_and_writes_content(tmp_path) -> None:
    path = tmp_path / "configs" / "config.yaml"

    write_yaml(str(path), {"task": "time_series_forecasting"})

    assert path.exists()
    assert "time_series_forecasting" in path.read_text(encoding="utf-8")


def test_read_yaml_reads_content(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("task: time_series_forecasting\n", encoding="utf-8")

    assert read_yaml(str(path)) == {"task": "time_series_forecasting"}


def test_read_yaml_returns_empty_dict_for_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert read_yaml(str(path)) == {}
