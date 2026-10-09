import hashlib
import json
from pathlib import Path

import pytest

from ragdocs.config import Settings
from ragdocs.errors import CorpusError
from ragdocs.ingestion import corpus_fingerprint, load_chunks


def make_corpus(root, texts=None):
    texts = texts or {"flat": "Vetores flat. " * 30, "ivf": "Índice IVF. " * 30}
    directory = root / "corpus"
    directory.mkdir()
    sources = []
    for name, text in texts.items():
        (directory / f"{name}.md").write_text(text, encoding="utf-8")
        sources.append(
            {
                "document_id": name,
                "title": f"Documento {name}",
                "file": f"{name}.md",
                "url": f"https://example.org/{name}",
                "revision": "frozen-sha",
                "license": "MIT",
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    (directory / "manifest.json").write_text(json.dumps({"schema_version": 1, "sources": sources}))
    return Settings(project_root=root, embedding_dimensions=2, chunk_size=100, chunk_overlap=20)


def test_split_preserves_origin_and_stable_identity(tmp_path):
    settings = make_corpus(tmp_path)
    chunks = load_chunks(settings)
    assert chunks == load_chunks(settings)
    assert len(chunks) > 2
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert 0 < len(chunk.text) <= 100
        assert chunk.url == f"https://example.org/{chunk.document_id}"
        assert chunk.revision == "frozen-sha"
        assert chunk.title == f"Documento {chunk.document_id}"
    assert corpus_fingerprint(settings) == corpus_fingerprint(settings)


@pytest.mark.parametrize("text", ["", "  \n\t"])
def test_empty_document_rejected(tmp_path, text):
    with pytest.raises(CorpusError):
        load_chunks(make_corpus(tmp_path, {"empty": text}))


def test_changed_content_rejected_even_without_manifest_edit(tmp_path):
    settings = make_corpus(tmp_path)
    (settings.corpus_dir / "flat.md").write_text("alterado")
    with pytest.raises(CorpusError):
        corpus_fingerprint(settings)


@pytest.mark.parametrize("file", ["../../secret.md", "/etc/passwd"])
def test_source_cannot_escape_corpus(tmp_path, file):
    settings = make_corpus(tmp_path)
    manifest_file = settings.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["sources"][0]["file"] = file
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(CorpusError):
        load_chunks(settings)


def test_invalid_overlap_rejected(tmp_path):
    settings = make_corpus(tmp_path).model_copy(update={"chunk_overlap": 100})
    with pytest.raises(CorpusError):
        load_chunks(settings)


def test_real_collection_six_sources_and_licenses():
    root = Path(__file__).resolve().parents[1]
    settings = Settings(project_root=root)
    chunks = load_chunks(settings)
    assert {chunk.document_id for chunk in chunks} == {
        "langchain-retrieval",
        "langchain-semantic-search",
        "faiss-readme",
        "faiss-flat",
        "faiss-ivf",
        "faiss-ivfpq",
    }
    assert any("IndexIVFPQ" in chunk.text for chunk in chunks)
    assert any("Finite context" in chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= 1200 for chunk in chunks)
    for license_file in (settings.corpus_dir / "licenses").glob("*.txt"):
        assert "Permission is hereby granted" in license_file.read_text()


@pytest.mark.parametrize("kind", ["duplicate", "schema", "json"])
def test_bad_manifest_is_safe_corpus_error(tmp_path, kind):
    settings = make_corpus(tmp_path)
    manifest_file = settings.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if kind == "duplicate":
        manifest["sources"].append(manifest["sources"][0])
    elif kind == "schema":
        manifest["schema_version"] = 2
    else:
        manifest_file.write_text("{broken")
        with pytest.raises(CorpusError):
            load_chunks(settings)
        return
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(CorpusError):
        load_chunks(settings)
