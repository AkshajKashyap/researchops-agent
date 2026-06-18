from researchops_agent.corpus.builder import build_corpus_manifest


def test_build_corpus_manifest_from_multiple_temp_docs(tmp_path) -> None:
    (tmp_path / "a.md").write_text("Ridge uses RMSE.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Graph kernels use accuracy.", encoding="utf-8")

    manifest, chunks = build_corpus_manifest(str(tmp_path))

    assert len(manifest.documents) == 2
    assert len(chunks) == 2
    assert manifest.total_pages == 2
    assert manifest.total_chunks == 2


def test_corpus_id_is_deterministic(tmp_path) -> None:
    (tmp_path / "a.md").write_text("Ridge uses RMSE.", encoding="utf-8")

    first, _ = build_corpus_manifest(str(tmp_path))
    second, _ = build_corpus_manifest(str(tmp_path))

    assert first.corpus_id == second.corpus_id


def test_manifest_counts_are_correct(tmp_path) -> None:
    (tmp_path / "a.md").write_text("Ridge " * 400, encoding="utf-8")

    manifest, chunks = build_corpus_manifest(str(tmp_path), chunk_size=100, overlap=20)

    assert manifest.total_pages == 1
    assert manifest.total_chunks == len(chunks)
    assert manifest.documents[0].num_chunks == len(chunks)
