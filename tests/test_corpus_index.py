from researchops_agent.corpus.builder import build_corpus_manifest
from researchops_agent.corpus.index import (
    load_corpus_chunks,
    load_corpus_manifest,
    load_corpus_metadata,
    save_corpus_index,
)
from researchops_agent.schemas.corpus import CorpusIndexMetadata


def test_save_and_load_corpus_index_parts(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Ridge uses RMSE.", encoding="utf-8")
    manifest, chunks = build_corpus_manifest(str(docs))
    metadata = CorpusIndexMetadata(
        corpus_id=manifest.corpus_id,
        root_path=str(docs),
        retriever="tfidf",
        chunk_size=1200,
        overlap=200,
        num_documents=len(manifest.documents),
        num_chunks=len(chunks),
    )
    index_dir = tmp_path / "index"

    save_corpus_index(str(index_dir), manifest, metadata, chunks)

    assert load_corpus_manifest(str(index_dir)) == manifest
    assert load_corpus_metadata(str(index_dir)) == metadata
    assert load_corpus_chunks(str(index_dir)) == chunks
