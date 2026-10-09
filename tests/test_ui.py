"""Verificação da experiência Streamlit sem provedor nem índice de produção."""

import json

import pytest
from pydantic import SecretStr
from streamlit.testing.v1 import AppTest

from ragdocs.config import Settings
from ragdocs.contracts import AnswerResult, SourceChunk
from ragdocs.errors import (
    AuthenticationError,
    CreditError,
    IndexIncompatibleError,
    InvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class FakeService:
    def __init__(self, result=None, error=None):
        self.result = result or answer()
        self.error = error
        self.questions = []
        self.creations = 0

    def factory(self, settings):
        self.creations += 1
        return self

    def ask(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.result


def answer():
    return AnswerResult(
        status="answered",
        answer="FAISS permite buscar vetores semelhantes. [faiss-1]",
        sources=[SourceChunk(
            chunk_id="faiss-1", text="Faiss is a library for similarity search.",
            document_id="faiss-readme", title="FAISS README",
            url="https://github.com/facebookresearch/faiss", revision="frozen-revision",
        )],
    )


def _app(settings, service):
    from ragdocs.ui import main

    main(service_factory=service.factory, settings=settings)


@pytest.fixture
def settings(tmp_path):
    config = Settings(project_root=tmp_path, api_key=SecretStr("test-secret-do-not-display"))
    config.index_dir.mkdir(parents=True)
    (config.index_dir / "index.faiss").write_bytes(b"fake-index")
    config.corpus_dir.mkdir()
    (config.corpus_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "sources": [{"title": "FAISS README", "url": "https://github.com/facebookresearch/faiss"}],
    }))
    return config


def app(settings, service=None):
    service = service or FakeService()
    return AppTest.from_function(_app, args=(settings, service)).run(), service


def test_answer_sources_and_cache(settings):
    at, service = app(settings)
    assert not at.exception
    assert service.creations == 0
    at.chat_input[0].set_value("O que é FAISS?").run()
    assert not at.exception
    assert service.questions == ["O que é FAISS?"]
    assert service.creations == 1
    assert "FAISS permite" in at.markdown[-1].value
    assert "FAISS README" in at.expander[0].label
    assert "similarity search" in at.code[0].value
    assert any(
        link.url == "https://github.com/facebookresearch/faiss" for link in at.get("link_button")
    )
    at.chat_input[0].set_value("Como buscar vetores?").run()
    assert service.creations == 1
    assert service.questions == ["O que é FAISS?", "Como buscar vetores?"]
    assert len(at.session_state["ragdocs_history"]) == 2


def test_index_change_invalidates_service(settings):
    at, service = app(settings)
    at.chat_input[0].set_value("Primeira pergunta").run()
    (settings.index_dir / "index.faiss").write_bytes(b"new-index-content")
    at.chat_input[0].set_value("Segunda pergunta").run()
    assert service.creations == 2


def test_history_is_capped_and_clearable(settings):
    at, service = app(settings)
    at.session_state["ragdocs_history"] = [
        {"question": f"Pergunta {i}", "result": answer()} for i in range(20)
    ]
    at.run()
    at.chat_input[0].set_value("Pergunta nova").run()
    history = at.session_state["ragdocs_history"]
    assert len(history) == 20
    assert history[0]["question"] == "Pergunta 1"
    assert history[-1]["question"] == "Pergunta nova"
    assert service.questions == ["Pergunta nova"]
    at.button[0].click().run()
    assert at.session_state["ragdocs_history"] == []
    assert not at.chat_message


def test_insufficient_context_is_not_technical_error(settings):
    result = AnswerResult(
        status="insufficient_context", answer="A coleção não sustenta essa resposta."
    )
    at, _ = app(settings, FakeService(result=result))
    at.chat_input[0].set_value("Qual a previsão do tempo?").run()
    assert at.info[0].value == result.answer
    assert not at.error
    assert not at.expander


@pytest.mark.parametrize("error,expected", [
    (AuthenticationError, "autenticação"),
    (CreditError, "Crédito"),
    (ProviderTimeoutError, "demorou"),
    (ProviderUnavailableError, "indisponível"),
    (IndexIncompatibleError, "não corresponde"),
    (InvalidResponseError, "validada"),
    (RuntimeError, "Não foi possível"),
])
def test_technical_errors_are_safe_and_do_not_enter_history(settings, error, expected):
    at, _ = app(settings, FakeService(error=error("test-secret-do-not-display sensitive response")))
    at.chat_input[0].set_value("O que é FAISS?").run()
    assert not at.exception
    assert expected in at.error[0].value
    assert "test-secret" not in at.error[0].value
    assert "sensitive response" not in at.error[0].value
    assert not at.info
    assert at.session_state["ragdocs_history"] == []


def test_missing_key_or_index_disables_questions(settings):
    no_key = settings.model_copy(update={"api_key": SecretStr("")})
    at, service = app(no_key)
    assert at.chat_input[0].disabled
    assert "OPENROUTER_API_KEY" in at.warning[0].value
    assert service.creations == 0
    (settings.index_dir / "index.faiss").unlink()
    at, _ = app(settings)
    assert at.chat_input[0].disabled
    assert "build-index" in at.warning[0].value


@pytest.mark.parametrize("question", ["   ", "x" * 1001])
def test_invalid_questions_never_reach_service(settings, question):
    at, service = app(settings)
    at.chat_input[0].set_value(question).run()
    assert at.error
    assert service.questions == []
    assert service.creations == 0


def test_missing_manifest_is_informational(settings):
    (settings.corpus_dir / "manifest.json").unlink()
    at, _ = app(settings)
    assert not at.exception
    assert any("Coleção indisponível" in caption.value for caption in at.caption)
