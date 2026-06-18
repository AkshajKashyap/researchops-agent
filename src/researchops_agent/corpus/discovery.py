from pathlib import Path

SUPPORTED_DOCUMENT_SUFFIXES = {".txt", ".md", ".pdf"}


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES


def discover_documents(root_path: str, recursive: bool = True) -> list[str]:
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(root_path)

    if root.is_file():
        if _is_supported(root) and not _is_hidden(root):
            return [str(root)]
        raise ValueError(f"No supported documents found under: {root_path}")

    pattern = "**/*" if recursive else "*"
    documents = [
        path
        for path in root.glob(pattern)
        if path.is_file() and _is_supported(path) and not _is_hidden(path.relative_to(root))
    ]
    sorted_documents = sorted(str(path) for path in documents)
    if not sorted_documents:
        raise ValueError(f"No supported documents found under: {root_path}")
    return sorted_documents
