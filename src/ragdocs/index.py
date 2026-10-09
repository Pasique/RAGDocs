"""FAISS CPU com integração LangChain e persistência nativa/JSON, sem pickle."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from ragdocs.config import Settings
from ragdocs.contracts import RetrievedChunk, SourceChunk
from ragdocs.errors import IndexIncompatibleError, IndexMissingError, ProviderError, QuestionError
from ragdocs.ingestion import corpus_fingerprint, load_chunks


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


def _identity(settings: Settings) -> dict:
    return {
        "schema_version": 1,
        "corpus_fingerprint": corpus_fingerprint(settings),
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "metric": "normalized_inner_product",
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_index(settings: Settings, embeddings: Embeddings | None = None) -> FaissRetriever:
    """Recarga sem embedding documental e sem construir um índice."""
    directory = settings.index_dir
    if not directory.exists():
        raise IndexMissingError("Índice ausente. Execute python -m ragdocs.cli build-index.")
    expected = _identity(settings)
    try:
        if not directory.is_dir():
            raise ValueError
        if not any(directory.iterdir()):
            raise IndexMissingError("Índice ausente. Execute python -m ragdocs.cli build-index.")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError
        if manifest["index_sha256"] != _sha(directory / "index.faiss"):
            raise ValueError
        if manifest["metadata_sha256"] != _sha(directory / "chunks.json"):
            raise ValueError
        metadata = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [SourceChunk.model_validate(item) for item in metadata]
        if not chunks or chunks != load_chunks(settings):
            raise ValueError
        native = faiss.read_index(str(directory / "index.faiss"))
        if (
            not isinstance(native, faiss.IndexFlatIP)
            or native.d != settings.embedding_dimensions
            or native.ntotal != len(chunks)
            or manifest["chunk_count"] != len(chunks)
        ):
            raise ValueError
        vectors = native.reconstruct_n(0, native.ntotal)
        if not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors.astype(np.float64), axis=1), 1.0, atol=1e-5
        ):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ValidationError):
        raise IndexIncompatibleError(
            "Índice incompatível ou inválido. Reconstrua com python -m ragdocs.cli build-index."
        ) from None
    return FaissRetriever(settings, _embeddings(settings, embeddings), native, chunks)


def build_index(settings: Settings, embeddings: Embeddings | None = None) -> FaissRetriever:
    """Reutilizar coleção compatível; construção explícita substitui índice incompatível."""
    identity = _identity(settings)
    try:
        return load_index(settings, embeddings)
    except (IndexMissingError, IndexIncompatibleError):
        pass
    chunks = load_chunks(settings)
    provider = _embeddings(settings, embeddings)
    vectors = _unit_vectors(
        provider.embed_documents([chunk.text for chunk in chunks]),
        len(chunks),
        settings.embedding_dimensions,
    )
    native = faiss.IndexFlatIP(settings.embedding_dimensions)
    native.add(vectors)
    directory = settings.index_dir
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # O manifesto é publicado por último. Uma interrupção será detectada na recarga.
        with tempfile.TemporaryDirectory(prefix=".build-", dir=directory) as temporary:
            staging = Path(temporary)
            faiss.write_index(native, str(staging / "index.faiss"))
            (staging / "chunks.json").write_text(
                json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False),
                encoding="utf-8",
            )
            manifest = {
                **identity,
                "chunk_count": len(chunks),
                "index_sha256": _sha(staging / "index.faiss"),
                "metadata_sha256": _sha(staging / "chunks.json"),
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
            )
            for name in ("index.faiss", "chunks.json", "manifest.json"):
                os.replace(staging / name, directory / name)
    except OSError:
        raise IndexIncompatibleError(
            "Não foi possível gravar o índice no diretório local."
        ) from None
    return FaissRetriever(settings, provider, native, chunks)
