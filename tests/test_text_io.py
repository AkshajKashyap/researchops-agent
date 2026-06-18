from researchops_agent.utils.text_io import write_text


def test_write_text_creates_parent_directories_and_writes_content(tmp_path) -> None:
    path = tmp_path / "nested" / "report.md"

    write_text(str(path), "# Report\n")

    assert path.read_text(encoding="utf-8") == "# Report\n"
