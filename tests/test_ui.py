"""Fluxo de PDFs e isolamento da sessão sem rede, parse ou índice de produção."""

import hashlib

import pytest
from pydantic import SecretStr
from streamlit.testing.v1 import AppTest

from ragdocs.config import Settings
from ragdocs.contracts import (
    AnswerResult,
    DocumentInfo,
    ExtractedCollection,
    PreparedCollection,
    SourceChunk,
    UploadedPDF,
)
from ragdocs.errors import (
    AuthenticationError,
    CreditError,
    DocumentError,
    InvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class FakeService:
    def __init__(self, result=None, error=None):
        self.result = result or answer()
        self.error = error
        self.questions = []
        self.creations = []

    def factory(self, settings, retriever):
        self.creations.append(retriever)
        return self

    def ask(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.result


class FakeRetriever:
    def search(self, question, k=4):
        return []


class FakeProcessor:
    def __init__(self, error=None, skipped_pages=0, duplicate_count=0):
        self.error = error
        self.calls = []
        self.skipped_pages = skipped_pages
        self.duplicate_count = duplicate_count

    def __call__(self, settings, uploads):
        self.calls.append(uploads)
        if self.error:
            raise self.error
        documents = []
        chunks = []
        for upload in uploads:
            identity = hashlib.sha256(upload.data).hexdigest()
            documents.append(
                DocumentInfo(
                    document_id=identity,
                    file_name=upload.name,
                    page_count=2,
                    text_pages=1,
                )
            )
            chunks.append(
                SourceChunk(
                    chunk_id=f"{identity}-2-0",
                    text="A biblioteca abre às nove horas.",
                    document_id=identity,
                    file_name=upload.name,
                    page=2,
                )
            )
        return PreparedCollection(
            collection=ExtractedCollection(
                fingerprint="synthetic-test-collection",
                documents=documents,
                chunks=chunks,
                skipped_pages=self.skipped_pages,
                duplicate_count=self.duplicate_count,
            ),
            retriever=FakeRetriever(),
        )


class FakeUploads:
    def __init__(self, selected=None):
        self.selected = selected if selected is not None else [pdf()]
        self.keys = []

    def __call__(self, key):
        self.keys.append(key)
        return self.selected if key == "ragdocs_uploads_0" else []


def pdf(name="biblioteca.pdf", content=b"synthetic-pdf-one"):
    return UploadedPDF(name=name, data=content)


def answer():
    return AnswerResult(
        status="answered",
        answer="A biblioteca abre às nove horas. [library-2]",
        sources=[
            SourceChunk(
                chunk_id="library-2",
                text="A biblioteca abre às nove horas.",
                document_id="library",
                file_name="biblioteca.pdf",
                page=2,
            )
        ],
    )


def _app(settings, service, processor, uploads):
    from ragdocs.ui import main

    main(
        service_factory=service.factory,
        settings=settings,
        processor=processor,
        upload_provider=uploads,
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(project_root=tmp_path, api_key=SecretStr("test-secret-do-not-display"))


def app(settings, service=None, processor=None, uploads=None, native=False):
    service = service or FakeService()
    processor = processor or FakeProcessor()
    uploads = None if native else (uploads or FakeUploads())
    at = AppTest.from_function(_app, args=(settings, service, processor, uploads)).run()
    return at, service, processor, uploads


def button(at, label):
    return next(widget for widget in at.button if widget.label == label)


def process(at):
    button(at, "Processar documentos").click().run()
    assert not at.exception
    return at


def test_requires_explicit_processing_then_reuses_session_index(settings):
    at, service, processor, _ = app(settings)
    assert not at.exception
    assert at.chat_input[0].disabled
    assert processor.calls == []
    assert service.creations == []
    at.run()
    assert processor.calls == []
    process(at)
    assert not at.chat_input[0].disabled
    assert len(processor.calls) == 1
    assert len(service.creations) == 1
    at.chat_input[0].set_value("Qual o horário da biblioteca?").run()
    assert not at.exception
    assert service.questions == ["Qual o horário da biblioteca?"]
    assert "nove horas" in at.markdown[-1].value
    assert "biblioteca.pdf — página 2" in at.expander[0].label
    assert "nove horas" in at.code[0].value
    assert not at.get("link_button")
    at.chat_input[0].set_value("Quando abre?").run()
    at.run()
    assert len(processor.calls) == 1
    assert len(service.creations) == 1
    assert len(at.session_state["ragdocs_history"]) == 2


def test_upload_multiple_documents_and_processing_summary(settings):
    selected = [pdf(), pdf("regulamento.pdf", b"different-content")]
    at, _, processor, _ = app(settings, uploads=FakeUploads(selected))
    process(at)
    assert processor.calls == [selected]
    assert "2 documento(s)" in at.success[0].value
    assert {text.value for text in at.text} >= {"biblioteca.pdf", "regulamento.pdf"}
    assert any("1 de 2 páginas" in caption.value for caption in at.caption)


def test_empty_pages_and_duplicates_are_disclosed(settings):
    at, _, _, _ = app(settings, processor=FakeProcessor(skipped_pages=2, duplicate_count=1))
    process(at)
    assert "2 página(s) sem texto" in at.warning[0].value
    assert any("1 arquivo(s) idêntico(s)" in caption.value for caption in at.caption)


@pytest.mark.parametrize(
    "replacement",
    [
        [pdf("biblioteca.pdf", b"new-content-same-name")],
        [pdf("renamed.pdf")],
        [],
    ],
)
def test_selection_change_immediately_invalidates_everything(settings, replacement):
    at, service, processor, uploads = app(settings)
    process(at)
    at.chat_input[0].set_value("Quando abre?").run()
    assert at.session_state["ragdocs_history"]
    uploads.selected = replacement
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled
    assert at.session_state["ragdocs_history"] == []
    assert "_ragdocs_prepared" not in at.session_state
    assert "_ragdocs_service" not in at.session_state
    assert not at.expander
    assert len(processor.calls) == 1
    assert len(service.creations) == 1


def test_configuration_change_invalidates_selection(settings):
    at, _, processor, _ = app(settings)
    process(at)
    object.__setattr__(settings, "embedding_model", "different-model")
    at.run()
    assert at.chat_input[0].disabled
    assert "_ragdocs_prepared" not in at.session_state
    assert len(processor.calls) == 1


def test_stale_question_event_cannot_query_changed_selection(settings):
    at, service, _, uploads = app(settings)
    process(at)
    uploads.selected = [pdf("outro.pdf", b"new-content")]
    at.chat_input[0].set_value("Pergunta enviada antes da troca").run()
    assert not at.exception
    assert service.questions == []
    assert at.chat_input[0].disabled
    assert at.session_state["ragdocs_history"] == []


def test_failed_reprocessing_never_restores_previous_collection(settings):
    at, service, processor, _ = app(settings)
    process(at)
    at.chat_input[0].set_value("Quando abre?").run()
    processor.error = DocumentError("test-secret-do-not-display sensitive document text")
    process(at)
    assert "PDFs" in at.error[0].value
    assert "sem senha" in at.error[0].value
    assert "test-secret" not in at.error[0].value
    assert "sensitive" not in at.error[0].value
    assert at.chat_input[0].disabled
    assert at.session_state["ragdocs_history"] == []
    assert "_ragdocs_prepared" not in at.session_state
    assert "_ragdocs_service" not in at.session_state
    assert len(service.creations) == 1


def test_factory_failure_does_not_leave_half_prepared_index(settings):
    class BadFactory(FakeService):
        def factory(self, settings, retriever):
            raise AuthenticationError("sensitive-provider-response")

    at, _, _, _ = app(settings, service=BadFactory())
    process(at)
    assert "autenticação" in at.error[0].value
    assert at.chat_input[0].disabled
    assert "_ragdocs_prepared" not in at.session_state


def test_history_is_capped_and_clear_does_not_remove_documents(settings):
    at, service, processor, _ = app(settings)
    process(at)
    prepared = at.session_state["_ragdocs_prepared"]
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
    button(at, "Limpar histórico").click().run()
    assert at.session_state["ragdocs_history"] == []
    assert not at.chat_message
    assert at.session_state["_ragdocs_prepared"] is prepared
    assert not at.chat_input[0].disabled
    assert len(processor.calls) == 1


def test_clear_documents_removes_uploads_index_service_and_history(settings):
    at, _, processor, uploads = app(settings)
    process(at)
    at.chat_input[0].set_value("Quando abre?").run()
    button(at, "Limpar documentos").click().run()
    assert not at.exception
    assert at.session_state["_ragdocs_uploads"] == []
    assert at.session_state["ragdocs_history"] == []
    assert "_ragdocs_prepared" not in at.session_state
    assert "_ragdocs_service" not in at.session_state
    assert at.chat_input[0].disabled
    assert button(at, "Processar documentos").disabled
    assert uploads.keys[-1] == "ragdocs_uploads_1"
    assert len(processor.calls) == 1


def test_two_sessions_keep_indices_uploads_and_history_isolated(settings):
    first, _, _, _ = app(settings)
    second, _, second_processor, _ = app(
        settings, uploads=FakeUploads([pdf("outro.pdf", b"other")])
    )
    process(first)
    first.chat_input[0].set_value("Quando abre?").run()
    assert second.chat_input[0].disabled
    assert second.session_state["ragdocs_history"] == []
    assert second_processor.calls == []
    process(second)
    assert first.session_state["_ragdocs_prepared"] is not second.session_state["_ragdocs_prepared"]
    first_document = first.session_state["_ragdocs_prepared"].collection.documents[0]
    second_document = second.session_state["_ragdocs_prepared"].collection.documents[0]
    assert first_document.file_name == "biblioteca.pdf"
    assert second_document.file_name == "outro.pdf"
    button(first, "Limpar documentos").click().run()
    assert not second.chat_input[0].disabled
    assert "_ragdocs_prepared" in second.session_state


def test_insufficient_context_is_not_technical_error(settings):
    result = AnswerResult(
        status="insufficient_context", answer="Os documentos não sustentam essa resposta."
    )
    at, _, _, _ = app(settings, service=FakeService(result=result))
    process(at)
    at.chat_input[0].set_value("Qual a previsão do tempo?").run()
    assert result.answer in [info.value for info in at.info]
    assert not at.error
    assert not at.expander


@pytest.mark.parametrize(
    "error,expected",
    [
        (AuthenticationError, "autenticação"),
        (CreditError, "Crédito"),
        (ProviderTimeoutError, "demorou"),
        (ProviderUnavailableError, "indisponível"),
        (InvalidResponseError, "validada"),
        (RuntimeError, "Não foi possível"),
    ],
)
def test_consultation_errors_are_safe_and_do_not_enter_history(settings, error, expected):
    service = FakeService(error=error("test-secret-do-not-display sensitive response"))
    at, _, _, _ = app(settings, service=service)
    process(at)
    at.chat_input[0].set_value("Quando abre?").run()
    assert not at.exception
    assert expected in at.error[0].value
    assert "test-secret" not in at.error[0].value
    assert "sensitive response" not in at.error[0].value
    assert at.session_state["ragdocs_history"] == []
    assert not at.chat_input[0].disabled


def test_missing_key_disables_processing_and_questions(settings):
    no_key = settings.model_copy(update={"api_key": SecretStr("")})
    at, service, processor, _ = app(no_key)
    assert at.chat_input[0].disabled
    assert button(at, "Processar documentos").disabled
    assert "OPENROUTER_API_KEY" in at.warning[0].value
    assert processor.calls == []
    assert service.creations == []


def test_missing_selection_disables_processing_and_questions(settings):
    at, service, processor, _ = app(settings, uploads=FakeUploads([]))
    assert at.chat_input[0].disabled
    assert button(at, "Processar documentos").disabled
    assert processor.calls == []
    assert service.creations == []


@pytest.mark.parametrize("question", ["   ", "x" * 1001])
def test_invalid_questions_never_reach_service(settings, question):
    at, service, _, _ = app(settings)
    process(at)
    at.chat_input[0].set_value(question).run()
    assert at.error
    assert service.questions == []


def test_processing_never_writes_upload_or_index_files(settings):
    before = list(settings.project_root.rglob("*"))
    at, _, _, _ = app(settings)
    process(at)
    at.chat_input[0].set_value("Quando abre?").run()
    assert list(settings.project_root.rglob("*")) == before
    assert at.session_state["_ragdocs_uploads"][0].data == b"synthetic-pdf-one"


def test_production_uploader_is_multiple_pdf_with_per_file_limit(settings):
    at, _, _, _ = app(settings, native=True)
    assert not at.exception
    uploader = at.get("file_uploader")[0]
    assert uploader.proto.type == [".pdf"]
    assert uploader.proto.max_upload_size_mb == 10
    assert uploader.proto.multiple_files
    assert at.chat_input[0].disabled
    assert any("Trechos dos PDFs serão enviados ao OpenRouter" in item.value for item in at.caption)
    assert not any("build-index" in item.value for item in at.warning)


def test_clear_documents_recreates_native_uploader(settings):
    at, _, _, _ = app(settings, native=True)
    original_id = at.get("file_uploader")[0].proto.id
    button(at, "Limpar documentos").click().run()
    assert not at.exception
    assert at.get("file_uploader")[0].proto.id != original_id
    assert at.session_state["_ragdocs_upload_generation"] == 1
    assert at.session_state["_ragdocs_uploads"] == []
