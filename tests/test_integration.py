import json
from pathlib import Path

import httpx
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
                "data": [{"index": i, "embedding": [2.0, 0.0]} for i in range(len(body["input"]))]
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
