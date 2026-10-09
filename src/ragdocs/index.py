"""FAISS CPU com integração LangChain; PDFs, metadados e índice somente em memória."""

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragdocs.config import Settings
from ragdocs.contracts import ExtractedCollection, PreparedCollection, RetrievedChunk, UploadedPDF
from ragdocs.errors import DocumentError, ProviderError, QuestionError
from ragdocs.ingestion import extract_collection


def _embeddings(settings: Settings, embeddings: Embeddings | None) -> Embeddings:
    if embeddings is None:
        from ragdocs.providers import OpenRouterEmbeddings

        return OpenRouterEmbeddings(settings)
    return embeddings


def _unit_vectors(vectors, count: int, dimensions: int) -> np.ndarray:
    try:
        array = np.asarray(vectors, dtype=np.float32)
        if array.shape != (count, dimensions) or not np.isfinite(array).all():
            raise ValueError
        norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
        if (norms <= 0).any() or not np.isfinite(norms).all():
            raise ValueError
        return np.ascontiguousarray(array / norms, dtype=np.float32)
    except (ValueError, TypeError, OverflowError):
        raise ProviderError(
            "Embeddings inválidos: confira as dimensões e o modelo configurado."
        ) from None


class FaissRetriever:
    def __init__(self, settings: Settings, embeddings: Embeddings, index, chunks):
        self.settings = settings
        self.chunks = chunks
        documents = {
            chunk.chunk_id: Document(
                page_content=chunk.text, metadata=chunk.model_dump(exclude={"text"})
            )
            for chunk in chunks
        }
        self.vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(documents),
            index_to_docstore_id={i: chunk.chunk_id for i, chunk in enumerate(chunks)},
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        self.embeddings = embeddings
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def search(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        if not question.strip() or len(question) > self.settings.max_question_chars:
            raise QuestionError("A pergunta deve conter entre 1 e 1.000 caracteres.")
        if k <= 0:
            raise QuestionError("A quantidade de trechos recuperados deve ser positiva.")
        vector = _unit_vectors(
            [self.embeddings.embed_query(question)], 1, self.settings.embedding_dimensions
        )[0]
        matches = self.vector_store.similarity_search_with_score_by_vector(
            vector.tolist(), k=min(k, len(self.chunks))
        )
        return [
            RetrievedChunk(chunk=self._by_id[doc.metadata["chunk_id"]], score=float(score))
            for doc, score in matches
        ]


def build_index(
    settings: Settings,
    collection: ExtractedCollection,
    embeddings: Embeddings | None = None,
) -> FaissRetriever:
    """Construir um índice somente em memória para uma coleção já validada."""
    if not collection.chunks:
        raise DocumentError("Os PDFs não produziram trechos para consulta.")
    provider = _embeddings(settings, embeddings)
    vectors = _unit_vectors(
        provider.embed_documents([chunk.text for chunk in collection.chunks]),
        len(collection.chunks),
        settings.embedding_dimensions,
    )
    native = faiss.IndexFlatIP(settings.embedding_dimensions)
    native.add(vectors)
    return FaissRetriever(settings, provider, native, collection.chunks)


def process_pdfs(
    settings: Settings,
    uploads: list[UploadedPDF],
    embeddings: Embeddings | None = None,
) -> PreparedCollection:
    """Validar todos os PDFs antes de inicializar o provedor e produzir embeddings."""
    collection = extract_collection(settings, uploads)
    return PreparedCollection(collection, build_index(settings, collection, embeddings))
