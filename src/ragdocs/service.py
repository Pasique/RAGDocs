"""Independent, source-grounded queries with schema and reference validation."""

import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from openrouter.errors import ResponseValidationError
from pydantic import ValidationError

from ragdocs.config import Settings
from ragdocs.contracts import AnswerResult, ModelAnswer, Retriever
from ragdocs.errors import (
    InvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuestionError,
)
from ragdocs.product_prompt import build_messages
from ragdocs.providers import create_chat_model, provider_error, retry_limit


class RagAnswerService:
    def __init__(self, settings: Settings, retriever: Retriever, model: Any = None):
        self.settings = settings
        self.retriever = retriever
        self._model = model

    def ask(self, question: str) -> AnswerResult:
        if not isinstance(question, str) or not question.strip():
            raise QuestionError("Digite uma pergunta não vazia.")
        question = question.strip()
        if len(question) > self.settings.max_question_chars:
            raise QuestionError(
                f"A pergunta deve ter até {self.settings.max_question_chars} caracteres."
            )
        retrieved = self.retriever.search(question, k=self.settings.top_k)
        if not retrieved:
            return AnswerResult(
                status="insufficient_context",
                answer="A coleção documental não contém trechos para responder a esta pergunta.",
            )
        if self._model is None:
            self._model = create_chat_model(self.settings).with_structured_output(
                ModelAnswer, method="json_schema", strict=True
            )
        messages = build_messages(question, retrieved)
        raw_answer = self._invoke(messages)
        try:
            parsed = ModelAnswer.model_validate(raw_answer)
        except (ValidationError, ValueError, TypeError):
            raise InvalidResponseError(
                "A resposta recebida não respeita o formato esperado."
            ) from None
        if not parsed.answer.strip() or re.search(r"https?://", parsed.answer, re.IGNORECASE):
            raise InvalidResponseError("A resposta recebida contém texto ou links inválidos.")
        source_map = {item.chunk.chunk_id: item.chunk for item in retrieved}
        if any(citation not in source_map for citation in parsed.citation_ids):
            raise InvalidResponseError("A resposta citou uma fonte que não foi recuperada.")
        if parsed.status == "answered" and not parsed.citation_ids:
            raise InvalidResponseError("A resposta documental não apresentou fontes.")
        if parsed.status == "insufficient_context" and parsed.citation_ids:
            raise InvalidResponseError(
                "A resposta de contexto insuficiente apresentou fontes inválidas."
            )
        sources = [source_map[chunk_id] for chunk_id in dict.fromkeys(parsed.citation_ids)]
        return AnswerResult(status=parsed.status, answer=parsed.answer.strip(), sources=sources)

    def _invoke(self, messages: list) -> Any:
        for attempt in range(retry_limit(self.settings) + 1):
            try:
                return self._model.invoke(messages)
            except (OutputParserException, ValidationError, ResponseValidationError):
                raise InvalidResponseError(
                    "A resposta recebida não respeita o formato esperado."
                ) from None
            except Exception as exc:
                safe_error = provider_error(exc)
                if isinstance(safe_error, (ProviderTimeoutError, ProviderUnavailableError)) and (
                    attempt < retry_limit(self.settings)
                ):
                    continue
                raise safe_error from None
        raise AssertionError("Unreachable retry state")


def create_service(settings: Settings) -> RagAnswerService:
    from ragdocs.index import load_index

    return RagAnswerService(settings, load_index(settings))
