"""OpenRouter adapters with bounded retries and safe public errors."""

import math
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter

from ragdocs.config import Settings
from ragdocs.errors import (
    AuthenticationError,
    CreditError,
    InvalidResponseError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def provider_error(exc: Exception) -> ProviderError:
    """Translate HTTP/SDK errors without exposing response bodies or credentials."""
    if isinstance(exc, ProviderError):
        return exc
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", getattr(response, "status_code", None))
    if status in (401, 403):
        return AuthenticationError("Chave OpenRouter inválida ou sem permissão.")
    if status == 402:
        return CreditError("Crédito ou limite da chave OpenRouter esgotado.")
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)) or status in (408, 504, 524):
        return ProviderTimeoutError("O OpenRouter excedeu o tempo de resposta. Tente depois.")
    if (
        isinstance(exc, httpx.TransportError)
        or status == 429
        or (isinstance(status, int) and status >= 500)
    ):
        return ProviderUnavailableError("OpenRouter indisponível ou com limite de requisições.")
    return ProviderError("Não foi possível concluir a chamada ao OpenRouter.")


def retry_limit(settings: Settings) -> int:
    """The product allows at most one retry, even with custom Settings."""
    return min(1, max(0, settings.max_retries))


class OpenRouterEmbeddings(Embeddings):
    """Embedding batches retain input order; injected clients support offline tests."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self._key = settings.require_api_key()
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise InvalidResponseError("Os textos para embeddings devem ser strings não vazias.")
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
            "dimensions": self.settings.embedding_dimensions,
            "encoding_format": "float",
        }
        if self._client is not None:
            return self._request(self._client, payload, len(texts))
        with httpx.Client(follow_redirects=False) as client:
            return self._request(client, payload, len(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _request(self, client: httpx.Client, payload: dict, count: int) -> list[list[float]]:
        for attempt in range(retry_limit(self.settings) + 1):
            try:
                response = client.post(
                    f"{self.settings.base_url.rstrip('/')}/embeddings",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                )
                response.raise_for_status()
            except (httpx.HTTPError, TimeoutError) as exc:
                safe_error = provider_error(exc)
                if isinstance(safe_error, (ProviderTimeoutError, ProviderUnavailableError)) and (
                    attempt < retry_limit(self.settings)
                ):
                    continue
                raise safe_error from None
            try:
                return self._validate(response.json(), count)
            except (ValueError, TypeError, KeyError, OverflowError):
                raise InvalidResponseError("O OpenRouter devolveu embeddings inválidos.") from None
        raise AssertionError("Unreachable retry state")

    def _validate(self, payload: Any, count: int) -> list[list[float]]:
        data = payload["data"]
        if not isinstance(data, list) or len(data) != count:
            raise ValueError("Wrong count")
        by_index = {}
        for item in data:
            index = item["index"]
            vector = item["embedding"]
            if type(index) is not int or not 0 <= index < count or index in by_index:
                raise ValueError("Invalid index")
            if not isinstance(vector, list) or len(vector) != self.settings.embedding_dimensions:
                raise ValueError("Wrong dimension")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
                raise ValueError("Non-finite vector")
            if not any(value != 0 for value in vector):
                raise ValueError("Zero vector cannot be normalized")
            by_index[index] = [float(value) for value in vector]
        return [by_index[index] for index in range(count)]


def create_chat_model(settings: Settings) -> ChatOpenRouter:
    """Disable SDK backoff; RagAnswerService owns the single allowed retry.

    In langchain-openrouter 0.2.9, max_retries represents a backoff time window,
    and zero alone leaves the SDK's default retries enabled. An explicit SDK
    client with retry_config=None disables that hidden retry layer.
    """
    key = settings.require_api_key()
    timeout_ms = int(settings.timeout_seconds * 1000)
    client = OpenRouter(
        api_key=key,
        server_url=settings.base_url,
        timeout_ms=timeout_ms,
        retry_config=None,
    )
    return ChatOpenRouter(
        model=settings.chat_model,
        api_key=key,
        base_url=settings.base_url,
        client=client,
        temperature=0,
        max_tokens=settings.max_output_tokens,
        timeout=timeout_ms,
        max_retries=0,
        openrouter_provider={"require_parameters": True},
        app_title="RAGDocs",
    )
