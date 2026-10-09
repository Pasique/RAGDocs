import faiss
import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from test_ingestion import upload

from ragdocs.config import Settings
from ragdocs.contracts import ExtractedCollection, UploadedPDF
from ragdocs.errors import DocumentError, ProviderError, QuestionError
from ragdocs.index import build_index, process_pdfs
from ragdocs.ingestion import extract_collection


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[30.0, 0.0] if "abacate" in text else [0.0, 5.0] for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return [7.0, 0.0] if "abacate" in text else [0.0, 9.0]


def prepared(tmp_path, embeddings=None):
    settings = Settings(
        project_root=tmp_path, embedding_dimensions=2, chunk_size=100, chunk_overlap=20
    )
    uploads = [upload("horta.pdf", ["abacate. " * 30]), upload("cozinha.pdf", ["batata. " * 30])]
    return settings, process_pdfs(settings, uploads, embeddings or FakeEmbeddings())


def test_real_faiss_normalization_ranking_metadata_and_no_files(tmp_path):
    provider = FakeEmbeddings()
    _, result = prepared(tmp_path, provider)
    retriever = result.retriever
    matches = retriever.search("abacate", k=2)
    assert len(matches) == 2
    assert all(item.chunk.file_name == "horta.pdf" and item.chunk.page == 1 for item in matches)
    assert all(item.score == pytest.approx(1.0) for item in matches)
    assert retriever.search("batata", k=1)[0].chunk.file_name == "cozinha.pdf"
    native = retriever.vector_store.index
    assert isinstance(native, faiss.IndexFlatIP)
    assert native.metric_type == faiss.METRIC_INNER_PRODUCT
    assert np.linalg.norm(native.reconstruct_n(0, native.ntotal), axis=1) == pytest.approx(
        np.ones(native.ntotal)
    )
    retriever.search("abacate")
    assert provider.document_calls == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("vector", [[0.0, 0.0], [1.0], [float("nan"), 0.0], [float("inf"), 0.0]])
def test_invalid_document_embedding_is_rejected_without_files(tmp_path, vector):
    provider = FakeEmbeddings()
    provider.embed_documents = lambda texts: [vector for _ in texts]
    with pytest.raises(ProviderError):
        prepared(tmp_path, provider)
    assert list(tmp_path.iterdir()) == []


def test_invalid_query_embedding_is_rejected(tmp_path):
    provider = FakeEmbeddings()
    _, result = prepared(tmp_path, provider)
    provider.embed_query = lambda text: [0.0, 0.0]
    with pytest.raises(ProviderError):
        result.retriever.search("abacate")


@pytest.mark.parametrize("question", ["", "  ", "a" * 1001])
def test_bad_question_does_not_call_provider(tmp_path, question):
    provider = FakeEmbeddings()
    _, result = prepared(tmp_path, provider)
    with pytest.raises(QuestionError):
        result.retriever.search(question)
    assert provider.query_calls == 0


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_does_not_call_provider(tmp_path, k):
    provider = FakeEmbeddings()
    _, result = prepared(tmp_path, provider)
    with pytest.raises(QuestionError):
        result.retriever.search("abacate", k=k)
    assert provider.query_calls == 0


def test_wrong_number_of_document_vectors_is_rejected(tmp_path):
    provider = FakeEmbeddings()
    provider.embed_documents = lambda texts: [[1.0, 0.0]]
    with pytest.raises(ProviderError):
        prepared(tmp_path, provider)


def test_k_larger_than_collection_returns_only_available_chunks(tmp_path):
    _, result = prepared(tmp_path)
    retriever = result.retriever
    matches = retriever.search("abacate", k=100)
    assert len(matches) == len(retriever.chunks)
    assert len({match.chunk.chunk_id for match in matches}) == len(matches)
    assert [match.score for match in matches] == sorted(
        (match.score for match in matches), reverse=True
    )


def test_invalid_last_pdf_prevents_any_embedding_for_batch(tmp_path):
    settings = Settings(project_root=tmp_path, embedding_dimensions=2)
    provider = FakeEmbeddings()
    with pytest.raises(DocumentError):
        process_pdfs(
            settings, [upload(), UploadedPDF(name="inválido.pdf", data=b"invalid")], provider
        )
    assert provider.document_calls == provider.query_calls == 0
    assert list(tmp_path.iterdir()) == []


def test_late_limit_prevents_any_embedding_for_batch(tmp_path):
    settings = Settings(project_root=tmp_path, max_collection_pages=1)
    provider = FakeEmbeddings()
    with pytest.raises(DocumentError):
        process_pdfs(settings, [upload(), upload("outro.pdf", ["diferente"])], provider)
    assert provider.document_calls == 0


def test_empty_collection_rejected_before_provider_initialization(tmp_path):
    with pytest.raises(DocumentError):
        build_index(
            Settings(project_root=tmp_path),
            ExtractedCollection(
                fingerprint="empty",
                documents=[],
                chunks=[],
            ),
        )


def test_independent_collections_do_not_share_index_or_sources(tmp_path):
    settings, first = prepared(tmp_path)
    second = process_pdfs(settings, [upload("novo.pdf", ["batata nova"])], FakeEmbeddings())
    assert first.retriever.vector_store.index is not second.retriever.vector_store.index
    assert second.collection.fingerprint != first.collection.fingerprint
    assert {chunk.file_name for chunk in second.retriever.chunks} == {"novo.pdf"}
    assert {chunk.file_name for chunk in first.retriever.chunks} == {"horta.pdf", "cozinha.pdf"}


def test_build_can_use_already_extracted_collection(tmp_path):
    settings = Settings(project_root=tmp_path, embedding_dimensions=2)
    collection = extract_collection(settings, [upload(texts=["abacate"])])
    assert build_index(settings, collection, FakeEmbeddings()).search("abacate")[0].chunk.page == 1
