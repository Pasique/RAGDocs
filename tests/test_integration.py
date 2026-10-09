import json
from pathlib import Path

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from ragdocs.config import Settings
from ragdocs.contracts import ModelAnswer, UploadedPDF
from ragdocs.index import process_pdfs
from ragdocs.providers import OpenRouterEmbeddings
from ragdocs.service import RagAnswerService


def test_http_embeddings_faiss_and_independent_pdf_answers(tmp_path):
    settings = Settings(project_root=tmp_path, api_key=SecretStr("offline"), embedding_dimensions=2)
    fixture = Path(__file__).parent / "fixtures" / "pdfs" / "manual_horta.pdf"
    calls = []

    def transport(request):
        body = json.loads(request.content)
        calls.append(body["input"])
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [2.0, 0.0] if "rega" in text.lower() else [0.0, 2.0]}
                    for i, text in enumerate(body["input"])
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(transport))
    embeddings = OpenRouterEmbeddings(settings, client)
    prepared = process_pdfs(
        settings, [UploadedPDF(name=fixture.name, data=fixture.read_bytes())], embeddings
    )
    assert prepared.retriever.search("Quando regar?")[0].score == 1
    assert not list(tmp_path.iterdir())  # Nenhum PDF, texto ou índice persistido.
    messages_seen = []

    class Model:
        def invoke(self, messages):
            messages_seen.append(messages)
            context = json.loads(messages[1].content)["context"]
            return ModelAnswer(
                status="answered",
                answer="A rega ocorre às 7h, três vezes por semana.",
                citation_ids=[context[0]["chunk_id"]],
            )

    service = RagAnswerService(settings, prepared.retriever, Model())
    first = service.ask("Quando regar?")
    second = service.ask("Qual o horário da rega?")
    assert first.sources[0].file_name == fixture.name
    assert first.sources[0].page == 1
    assert second.sources == first.sources
    assert all(len(messages) == 2 for messages in messages_seen)
    assert "Quando regar?" not in messages_seen[-1][1].content
    assert len(calls) == 4  # Indexação, busca de sanidade e duas perguntas.
    assert not list(tmp_path.iterdir())
    client.close()


class SessionEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[1.0, 0.0] if "rega" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class SessionModel:
    def invoke(self, messages):
        context = json.loads(messages[1].content)["context"]
        return ModelAnswer(
            status="answered",
            answer="A rega ocorre às 7h, três vezes por semana.",
            citation_ids=[context[0]["chunk_id"]],
        )


def _session_app(settings, uploads, embeddings, model):
    from ragdocs.index import process_pdfs
    from ragdocs.service import RagAnswerService
    from ragdocs.ui import main

    main(
        settings=settings,
        upload_provider=lambda key: uploads,
        processor=lambda config, files: process_pdfs(config, files, embeddings),
        service_factory=lambda config, retriever: RagAnswerService(
            config, retriever, model
        ),
    )


def test_ui_uses_real_pdf_pipeline_and_reuses_index(tmp_path):
    from streamlit.testing.v1 import AppTest

    fixture = Path(__file__).parent / "fixtures" / "pdfs" / "manual_horta.pdf"
    uploads = [UploadedPDF(name=fixture.name, data=fixture.read_bytes())]
    settings = Settings(project_root=tmp_path, api_key=SecretStr("offline"), embedding_dimensions=2)
    embeddings = SessionEmbeddings()
    at = AppTest.from_function(
        _session_app, args=(settings, uploads, embeddings, SessionModel())
    ).run()
    assert not at.exception
    next(button for button in at.button if button.label == "Processar documentos").click().run()
    assert not at.exception
    assert not at.error
    at.chat_input[0].set_value("Quando regar?").run()
    assert not at.exception
    assert "manual_horta.pdf — página 1" in at.expander[0].label
    assert "7h" in at.code[0].value
    at.run()
    assert embeddings.document_calls == 1
    uploads[:] = [UploadedPDF(name="outro.pdf", data=b"invalid")]
    at.run()
    assert at.chat_input[0].disabled
    assert not at.expander
    assert at.session_state["ragdocs_history"] == []
    next(button for button in at.button if button.label == "Processar documentos").click().run()
    assert at.error
    assert at.chat_input[0].disabled
    assert embeddings.document_calls == 1
    assert not list(tmp_path.iterdir())
