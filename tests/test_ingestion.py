import pytest

from researchops_agent.ingestion.loaders import load_document, load_markdown, load_text


def test_load_text(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Hello   world\n\n\nNext line", encoding="utf-8")

    pages = load_text(str(path))

    assert len(pages) == 1
    assert pages[0].source.path == str(path)
    assert pages[0].source.source_type == "text"
    assert pages[0].page_number is None
    assert pages[0].text == "Hello world\n\nNext line"


def test_load_markdown(tmp_path) -> None:
    path = tmp_path / "paper.md"
    path.write_text("# Title\n\nSome   markdown", encoding="utf-8")

    pages = load_markdown(str(path))

    assert len(pages) == 1
    assert pages[0].source.path == str(path)
    assert pages[0].source.source_type == "markdown"
    assert pages[0].text == "# Title\n\nSome markdown"


def test_load_document_rejects_unsupported_file_type(tmp_path) -> None:
    path = tmp_path / "paper.docx"
    path.write_text("Nope", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(str(path))


def test_load_document_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(str(tmp_path / "missing.txt"))
