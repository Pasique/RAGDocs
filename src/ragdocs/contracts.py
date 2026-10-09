"""Contratos públicos v1 para ingestão, recuperação e apresentação."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class SourceChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: str
    text: str
    document_id: str
    title: str
    url: str
    revision: str


class RetrievedChunk(BaseModel):
    chunk: SourceChunk
    score: float


class AnswerResult(BaseModel):
    status: Literal["answered", "insufficient_context"]
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)


class ModelAnswer(BaseModel):
    """Resposta estruturada do modelo; URLs sempre vêm do corpus."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_context"]
    answer: str = Field(min_length=1)
    citation_ids: list[str]


class Retriever(Protocol):
    def search(self, question: str, k: int = 4) -> list[RetrievedChunk]: ...


class AnswerService(Protocol):
    def ask(self, question: str) -> AnswerResult: ...
