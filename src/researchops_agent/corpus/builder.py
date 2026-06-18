import hashlib
import json

from researchops_agent.corpus.discovery import discover_documents
from researchops_agent.ingestion.chunking import chunk_pages
from researchops_agent.ingestion.loaders import load_document
from researchops_agent.schemas.corpus import CorpusDocument, CorpusManifest
from researchops_agent.schemas.document import DocumentChunk


def _corpus_id(root_path: str, document_paths: list[str]) -> str:
    payload = {"root_path": root_path, "documents": document_paths}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"corpus_{digest[:12]}"


def build_corpus_manifest(
    root_path: str,
    recursive: bool = True,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> tuple[CorpusManifest, list[DocumentChunk]]:
    document_paths = discover_documents(root_path, recursive=recursive)
    documents: list[CorpusDocument] = []
    all_chunks: list[DocumentChunk] = []

    for path in document_paths:
        try:
            pages = load_document(path)
            chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        except Exception as exc:
            raise ValueError(f"Failed to process document {path}: {exc}") from exc

        source = pages[0].source if pages else None
        documents.append(
            CorpusDocument(
                path=path,
                source_type=source.source_type if source else "unknown",
                title=source.title if source else None,
                num_pages=len(pages),
                num_chunks=len(chunks),
            )
        )
        all_chunks.extend(chunks)

    manifest = CorpusManifest(
        corpus_id=_corpus_id(root_path, document_paths),
        root_path=root_path,
        documents=documents,
        total_pages=sum(document.num_pages for document in documents),
        total_chunks=len(all_chunks),
    )
    return manifest, all_chunks
