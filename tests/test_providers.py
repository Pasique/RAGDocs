import json

import httpx
import pytest
from pydantic import SecretStr

from ragdocs.config import Settings
from ragdocs.errors import (
    AuthenticationError,
    ConfigurationError,
    CreditError,
    InvalidResponseError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ragdocs.providers import OpenRouterEmbeddings, create_chat_model, provider_error


@pytest.fixture
def settings():
    return Settings(api_key=SecretStr("test-key-not-real"), embedding_dimensions=2)


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_batch_preserves_order_and_request_configuration(settings):
    def handler(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/embeddings"
        assert request.method == "POST"
        assert request.headers["Authorization"] == "Bearer test-key-not-real"
        assert request.extensions["timeout"]["read"] == 30
        assert json.loads(request.content) == {
            "model": "openai/text-embedding-3-small",
            "input": ["primeiro", "segundo"],
            "dimensions": 2,
            "encoding_format": "float",
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 2]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            },
        )

    with client_for(handler) as client:
        assert OpenRouterEmbeddings(settings, client).embed_documents(["primeiro", "segundo"]) == [
            [1.0, 0.0],
            [0.0, 2.0],
        ]


def test_query_and_empty_batch(settings):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2]}]})

    with client_for(handler) as client:
        adapter = OpenRouterEmbeddings(settings, client)
        assert adapter.embed_documents([]) == []
        assert adapter.embed_query("consulta") == [1, 2]
    assert calls == [["consulta"]]


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 1, "embedding": [1, 2]}],
        [{"index": True, "embedding": [1, 2]}],
        [{"index": 0, "embedding": [1]}],
        [{"index": 0, "embedding": [True, 2]}],
        [{"index": 0, "embedding": ["1", 2]}],
        [{"index": 0, "embedding": [0, 0]}],
        [{"embedding": [1, 2]}],
        ["malformed"],
    ],
)
def test_invalid_embedding_response_not_retried(settings, data):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": data})

    with client_for(handler) as client:
        with pytest.raises(InvalidResponseError):
            OpenRouterEmbeddings(settings, client).embed_query("consulta")
    assert len(calls) == 1


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -float("inf")])
def test_non_finite_vectors(settings, value):
    with client_for(lambda request: httpx.Response(200)) as client:
        adapter = OpenRouterEmbeddings(settings, client)
        with pytest.raises(ValueError):
            adapter._validate({"data": [{"index": 0, "embedding": [value, 1]}]}, 1)


def test_duplicate_indices_rejected(settings):
    with client_for(
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1, 2]},
                    {"index": 0, "embedding": [2, 1]},
                ]
            },
        )
    ) as client:
        with pytest.raises(InvalidResponseError):
            OpenRouterEmbeddings(settings, client).embed_documents(["um", "dois"])


@pytest.mark.parametrize(
    "status,error,calls_expected",
    [
        (401, AuthenticationError, 1),
        (403, AuthenticationError, 1),
        (402, CreditError, 1),
        (429, ProviderUnavailableError, 2),
        (503, ProviderUnavailableError, 2),
        (504, ProviderTimeoutError, 2),
        (400, ProviderError, 1),
    ],
)
def test_http_errors_are_safe_and_retries_bounded(settings, status, error, calls_expected):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="sensitive body test-key-not-real")

    with client_for(handler) as client:
        with pytest.raises(error) as caught:
            OpenRouterEmbeddings(settings, client).embed_query("consulta")
    assert len(calls) == calls_expected
    assert "sensitive" not in str(caught.value)
    assert "test-key-not-real" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "exception,error",
    [
        (httpx.ReadTimeout, ProviderTimeoutError),
        (httpx.ConnectError, ProviderUnavailableError),
    ],
)
def test_network_failure_retried_once(settings, exception, error):
    calls = []

    def handler(request):
        calls.append(request)
        raise exception("private failure", request=request)

    with client_for(handler) as client:
        with pytest.raises(error):
            OpenRouterEmbeddings(settings, client).embed_query("consulta")
    assert len(calls) == 2


def test_transient_failure_then_success(settings):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2]}]})

    with client_for(handler) as client:
        assert OpenRouterEmbeddings(settings, client).embed_query("consulta") == [1, 2]
    assert len(calls) == 2


def test_retry_setting_cannot_exceed_product_limit(settings):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    with client_for(handler) as client:
        with pytest.raises(ProviderUnavailableError):
            OpenRouterEmbeddings(
                settings.model_copy(update={"max_retries": 50}), client
            ).embed_query("q")
    assert len(calls) == 2


def test_key_required_and_input_validation(settings):
    with pytest.raises(ConfigurationError):
        OpenRouterEmbeddings(Settings())
    with client_for(lambda request: pytest.fail("should not call")) as client:
        with pytest.raises(InvalidResponseError):
            OpenRouterEmbeddings(settings, client).embed_documents([""])


def test_invalid_json_not_retried(settings):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text="private text")

    with client_for(handler) as client:
        with pytest.raises(InvalidResponseError):
            OpenRouterEmbeddings(settings, client).embed_query("q")
    assert len(calls) == 1


def test_chat_configuration_disables_sdk_retry_and_uses_milliseconds(settings):
    model = create_chat_model(settings)
    assert model.model == settings.chat_model
    assert model.temperature == 0
    assert model.max_tokens == 700
    assert model.request_timeout == 30000
    assert model.max_retries == 0
    assert model.openrouter_provider == {"require_parameters": True}
    assert model.client.sdk_configuration.retry_config is None
    assert model.client.sdk_configuration.timeout_ms == 30000


def test_sdk_status_mapping_does_not_expose_body():
    class SDKError(Exception):
        status_code = 402

    assert isinstance(provider_error(SDKError("private body")), CreditError)
