import json
from pathlib import Path

from researchops_agent.schemas.corpus import CorpusIndexMetadata, CorpusManifest
from researchops_agent.schemas.document import DocumentChunk


def save_corpus_index(
    index_dir: str,
    manifest: CorpusManifest,
    metadata: CorpusIndexMetadata,
    chunks: list[DocumentChunk],
) -> None:
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    (index_path / "manifest.json").write_text(
        json.dumps(manifest.model_dump(), indent=2) + "\n",
        encoding="utf-8",
    )
    (index_path / "metadata.json").write_text(
        json.dumps(metadata.model_dump(), indent=2) + "\n",
        encoding="utf-8",
    )
    (index_path / "chunks.json").write_text(
        json.dumps([chunk.model_dump() for chunk in chunks], indent=2) + "\n",
        encoding="utf-8",
    )


def load_corpus_manifest(index_dir: str) -> CorpusManifest:
    data = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
    return CorpusManifest.model_validate(data)


def load_corpus_metadata(index_dir: str) -> CorpusIndexMetadata:
    data = json.loads((Path(index_dir) / "metadata.json").read_text(encoding="utf-8"))
    return CorpusIndexMetadata.model_validate(data)


def load_corpus_chunks(index_dir: str) -> list[DocumentChunk]:
    data = json.loads((Path(index_dir) / "chunks.json").read_text(encoding="utf-8"))
    return [DocumentChunk.model_validate(item) for item in data]
