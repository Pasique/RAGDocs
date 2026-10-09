"""Contratos v2: arquivos escolhidos pelo usuário, fontes por página e índice em memória."""

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class UploadedPDF(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str = Field(min_length=1)
    data: bytes = Field(repr=False, exclude=True)


class SourceChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: str
    text: str
    document_id: str
    file_name: str
    page: int = Field(ge=1)


class DocumentInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: str
    file_name: str
    page_count: int = Field(ge=1)
    text_pages: int = Field(ge=1)


class ExtractedCollection(BaseModel):
    fingerprint: str
    documents: list[DocumentInfo]
    chunks: list[SourceChunk]
    skipped_pages: int = 0
    duplicate_count: int = 0


class RetrievedChunk(BaseModel):
    chunk: SourceChunk
    score: float


class AnswerResult(BaseModel):
    status: Literal["answered", "insufficient_context"]
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)


class ModelAnswer(BaseModel):
    """Resposta estruturada; arquivo, página e texto sempre vêm dos PDFs."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_context"]
    answer: str = Field(min_length=1)
    citation_ids: list[str]


class Retriever(Protocol):
    def search(self, question: str, k: int = 4) -> list[RetrievedChunk]: ...


class AnswerService(Protocol):
    def ask(self, question: str) -> AnswerResult: ...


@dataclass(frozen=True)
class PreparedCollection:
    collection: ExtractedCollection
    retriever: Retriever
