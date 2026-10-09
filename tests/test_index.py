import json

import faiss
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from test_ingestion import make_corpus

from ragdocs.errors import IndexIncompatibleError, IndexMissingError, ProviderError, QuestionError
from ragdocs.index import build_index, load_index


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[30.0, 0.0] if "flat" in text else [0.0, 5.0] for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return [7.0, 0.0] if "flat" in text else [0.0, 9.0]


def test_real_faiss_normalization_ranking_and_json_roundtrip(tmp_path):
    settings = make_corpus(tmp_path)
    embeddings = FakeEmbeddings()
    retriever = build_index(settings, embeddings)
    flat = retriever.search("flat", k=2)
    assert len(flat) == 2
    assert all(item.chunk.document_id == "flat" for item in flat)
    assert all(item.score == pytest.approx(1.0) for item in flat)
    assert retriever.search("ivf", k=1)[0].chunk.document_id == "ivf"
    native = faiss.read_index(str(settings.index_dir / "index.faiss"))
    assert native.metric_type == faiss.METRIC_INNER_PRODUCT
    assert np.linalg.norm(native.reconstruct_n(0, native.ntotal), axis=1) == pytest.approx(
        np.ones(native.ntotal)
    )
    loaded = load_index(settings, embeddings)
    assert loaded.search("flat", k=2) == flat
    reused = build_index(settings, embeddings)
    assert reused.search("flat", k=2) == flat
    assert embeddings.document_calls == 1
    assert {p.name for p in settings.index_dir.iterdir()} == {
        "index.faiss",
        "chunks.json",
        "manifest.json",
    }


def test_load_absent_never_calls_embeddings(tmp_path):
    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    with pytest.raises(IndexMissingError):
        load_index(settings, provider)
    assert provider.document_calls == provider.query_calls == 0


@pytest.mark.parametrize(
    "update",
    [
        {"embedding_model": "another/model"},
        {"embedding_dimensions": 3},
        {"chunk_size": 200},
        {"chunk_overlap": 10},
    ],
)
def test_load_rejects_incompatible_settings_without_embedding(tmp_path, update):
    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    build_index(settings, provider)
    with pytest.raises(IndexIncompatibleError):
        load_index(settings.model_copy(update=update), provider)
    assert provider.document_calls == 1


@pytest.mark.parametrize("name", ["index.faiss", "chunks.json", "manifest.json"])
def test_corruption_is_safe_failure(tmp_path, name):
    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    build_index(settings, provider)
    (settings.index_dir / name).write_bytes(b"corrompido")
    with pytest.raises(IndexIncompatibleError):
        load_index(settings, provider)
    assert provider.document_calls == 1


def test_explicit_build_replaces_incompatible_index(tmp_path):
    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    build_index(settings, provider)
    changed = settings.model_copy(update={"embedding_model": "another/model"})
    build_index(changed, provider)
    assert provider.document_calls == 2
    assert json.loads((settings.index_dir / "manifest.json").read_text())["embedding_model"] == (
        "another/model"
    )


@pytest.mark.parametrize("vector", [[0.0, 0.0], [1.0], [float("nan"), 0.0], [float("inf"), 0.0]])
def test_invalid_document_embedding_is_not_persisted(tmp_path, vector):
    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    provider.embed_documents = lambda texts: [vector for _ in texts]
    with pytest.raises(ProviderError):
        build_index(settings, provider)
    assert not settings.index_dir.exists()


def test_invalid_query_embedding_is_rejected(tmp_path):
    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    retriever = build_index(settings, provider)
    provider.embed_query = lambda text: [0.0, 0.0]
    with pytest.raises(ProviderError):
        retriever.search("flat")


@pytest.mark.parametrize("question", ["", "  ", "a" * 1001])
def test_bad_question_does_not_call_provider(tmp_path, question):
    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    retriever = build_index(settings, provider)
    with pytest.raises(QuestionError):
        retriever.search(question)
    assert provider.query_calls == 0


def test_changed_corpus_requires_explicit_rebuild(tmp_path):
    import hashlib

    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    build_index(settings, provider)
    text = "flat atualizado. " * 30
    (settings.corpus_dir / "flat.md").write_text(text)
    manifest_file = settings.corpus_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["sources"][0]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(IndexIncompatibleError):
        load_index(settings, provider)
    assert provider.document_calls == 1
    build_index(settings, provider)
    assert provider.document_calls == 2


def test_wrong_number_of_document_vectors_is_rejected(tmp_path):
    settings = make_corpus(tmp_path)
    provider = FakeEmbeddings()
    provider.embed_documents = lambda texts: [[1.0, 0.0]]
    with pytest.raises(ProviderError):
        build_index(settings, provider)


def test_k_larger_than_collection_returns_only_available_chunks(tmp_path):
    settings = make_corpus(tmp_path)
    retriever = build_index(settings, FakeEmbeddings())
    matches = retriever.search("flat", k=100)
    assert len(matches) == len(retriever.chunks)
    assert len({match.chunk.chunk_id for match in matches}) == len(matches)
    assert [match.score for match in matches] == sorted(
        (match.score for match in matches), reverse=True
    )
