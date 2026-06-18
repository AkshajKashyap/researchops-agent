from researchops_agent.corpus.builder import build_corpus_manifest
from researchops_agent.corpus.index import (
    load_corpus_chunks,
    load_corpus_manifest,
    load_corpus_metadata,
    save_corpus_index,
)
from researchops_agent.retrieval.factory import build_retriever
from researchops_agent.schemas.corpus import CorpusIndexMetadata, CorpusSearchResult


def build_index_for_corpus(
    root_path: str,
    index_dir: str,
    retriever_kind: str = "tfidf",
    recursive: bool = True,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> CorpusIndexMetadata:
    build_retriever(retriever_kind)
    manifest, chunks = build_corpus_manifest(
        root_path,
        recursive=recursive,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    metadata = CorpusIndexMetadata(
        corpus_id=manifest.corpus_id,
        root_path=root_path,
        retriever=retriever_kind,
        chunk_size=chunk_size,
        overlap=overlap,
        num_documents=len(manifest.documents),
        num_chunks=len(chunks),
    )
    save_corpus_index(index_dir, manifest, metadata, chunks)
    return metadata


def search_corpus(
    index_dir: str,
    query: str,
    top_k: int = 5,
    retriever_kind: str | None = None,
) -> CorpusSearchResult:
    manifest = load_corpus_manifest(index_dir)
    metadata = load_corpus_metadata(index_dir)
    chunks = load_corpus_chunks(index_dir)
    selected_retriever = retriever_kind or metadata.retriever
    retriever = build_retriever(selected_retriever)
    retriever.fit(chunks)
    retrieval = retriever.search(query, top_k=top_k)
    return CorpusSearchResult(
        query=query,
        corpus_id=manifest.corpus_id,
        results=retrieval.results,
    )
