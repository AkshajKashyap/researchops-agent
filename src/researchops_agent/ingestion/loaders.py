from pathlib import Path
import re

from researchops_agent.schemas.document import DocumentPage, DocumentSource


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _path_or_raise(path: str) -> Path:
    document_path = Path(path)
    if not document_path.exists():
        raise FileNotFoundError(path)
    return document_path


def load_text(path: str) -> list[DocumentPage]:
    document_path = _path_or_raise(path)
    source = DocumentSource(path=str(document_path), source_type="text")
    text = _normalize_text(document_path.read_text(encoding="utf-8"))
    return [DocumentPage(source=source, text=text)]


def load_markdown(path: str) -> list[DocumentPage]:
    document_path = _path_or_raise(path)
    source = DocumentSource(path=str(document_path), source_type="markdown")
    text = _normalize_text(document_path.read_text(encoding="utf-8"))
    return [DocumentPage(source=source, text=text)]


def load_pdf(path: str) -> list[DocumentPage]:
    document_path = _path_or_raise(path)
    source = DocumentSource(path=str(document_path), source_type="pdf")

    from pypdf import PdfReader

    reader = PdfReader(str(document_path))
    pages: list[DocumentPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        pages.append(DocumentPage(source=source, page_number=page_number, text=text))
    return pages


def load_document(path: str) -> list[DocumentPage]:
    document_path = _path_or_raise(path)
    extension = document_path.suffix.lower()

    if extension == ".txt":
        return load_text(str(document_path))
    if extension == ".md":
        return load_markdown(str(document_path))
    if extension == ".pdf":
        return load_pdf(str(document_path))

    raise ValueError(f"Unsupported document type: {extension or '<no extension>'}")
