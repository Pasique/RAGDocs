import hashlib
import json

import httpx
from pydantic import SecretStr

from ragdocs.config import Settings
from ragdocs.contracts import ModelAnswer
from ragdocs.index import build_index, load_index
from ragdocs.providers import OpenRouterEmbeddings
from ragdocs.service import RagAnswerService


def test_http_embeddings_faiss_and_independent_answers(tmp_path):
    settings = Settings(project_root=tmp_path, api_key=SecretStr("offline"), embedding_dimensions=2)
    settings.corpus_dir.mkdir()
    content = (
        "RAG recupera documentos antes de gerar respostas.\n```python\nraise RuntimeError()\n```"
    )
    (settings.corpus_dir / "doc.md").write_text(content)
    (settings.corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "document_id": "doc",
                        "title": "Guia RAG",
                        "file": "doc.md",
                        "url": "https://example.org/guide",
                        "revision": "abc",
                        "license": "MIT",
                        "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    }
                ],
            }
        )
    )
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
    built = build_index(settings, embeddings)
    assert built.search("O que é RAG?")[0].score == 1
    assert not list(settings.index_dir.glob("*.pkl"))
    retriever = load_index(settings, embeddings)
    messages_seen = []

    class Model:
        def invoke(self, messages):
            messages_seen.append(messages)
            context = json.loads(messages[1].content)["context"]
            return ModelAnswer(
                status="answered",
                answer="Recupera antes de gerar.",
                citation_ids=[context[0]["chunk_id"]],
            )

    service = RagAnswerService(settings, retriever, Model())
    first = service.ask("O que é RAG?")
    second = service.ask("Como são geradas respostas?")
    assert first.sources[0].url == "https://example.org/guide"
    assert second.sources == first.sources
    assert all(len(messages) == 2 for messages in messages_seen)
    assert "O que é RAG?" not in messages_seen[-1][1].content
    assert len(calls) == 4  # Indexação, busca de sanidade e duas perguntas.
    client.close()
