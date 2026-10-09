"""Leitura de corpus fixo, validação de origem e divisão determinística."""

import hashlib
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragdocs.config import Settings
from ragdocs.contracts import SourceChunk
from ragdocs.errors import CorpusError


class CorpusSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    file: str = Field(min_length=1)
    url: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _read_sources(settings: Settings) -> list[tuple[CorpusSource, str]]:
    root = settings.corpus_dir.resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or not manifest["sources"]:
            raise ValueError
        sources = [CorpusSource.model_validate(item) for item in manifest["sources"]]
        if len({source.document_id for source in sources}) != len(sources):
            raise ValueError
        result = []
        for source in sources:
            path = (root / source.file).resolve()
            if not path.is_relative_to(root) or not source.url.startswith("https://"):
                raise ValueError
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != source.sha256:
                raise ValueError
            text = content.decode("utf-8")
            if not text.strip():
                raise ValueError
            result.append((source, text))
        return result
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, ValidationError):
        raise CorpusError(
            "Corpus ausente, vazio ou inválido. Confira os arquivos e o manifesto da coleção."
        ) from None


def corpus_fingerprint(settings: Settings) -> str:
    """Identidade da coleção validada; inclui conteúdo e todos os metadados de origem."""
    data = [source.model_dump() for source, _ in _read_sources(settings)]
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_chunks(settings: Settings) -> list[SourceChunk]:
    """Dividir texto; exemplos de código são tratados somente como texto."""
    if settings.chunk_size <= 0 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise CorpusError("Parâmetros de divisão inválidos: confira tamanho e sobreposição.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for source, text in _read_sources(settings):
        for position, part in enumerate(splitter.split_text(text)):
            identity = hashlib.sha256(part.encode("utf-8")).hexdigest()[:12]
            chunks.append(
                SourceChunk(
                    chunk_id=f"{source.document_id}:{position:04d}:{identity}",
                    text=part,
                    document_id=source.document_id,
                    title=source.title,
                    url=source.url,
                    revision=source.revision,
                )
            )
    if not chunks:
        raise CorpusError("A coleção não produziu trechos para indexação.")
    return chunks
