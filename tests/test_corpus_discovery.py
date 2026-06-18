import pytest

from researchops_agent.corpus.discovery import discover_documents


def test_discover_documents_finds_supported_files_in_temp_directory(tmp_path) -> None:
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("C", encoding="utf-8")

    documents = discover_documents(str(tmp_path), recursive=False)

    assert documents == [str(tmp_path / "a.md"), str(tmp_path / "b.txt")]


def test_discover_documents_recursive_search_works(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "paper.md").write_text("Paper", encoding="utf-8")

    assert discover_documents(str(tmp_path)) == [str(nested / "paper.md")]


def test_discover_documents_unsupported_directory_raises_value_error(tmp_path) -> None:
    (tmp_path / "data.csv").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="No supported documents"):
        discover_documents(str(tmp_path))


def test_discover_documents_missing_path_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        discover_documents("missing/path")


def test_discover_documents_ignores_hidden_files_and_directories(tmp_path) -> None:
    (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "nested.md").write_text("hidden", encoding="utf-8")
    (tmp_path / "visible.md").write_text("visible", encoding="utf-8")

    assert discover_documents(str(tmp_path)) == [str(tmp_path / "visible.md")]
