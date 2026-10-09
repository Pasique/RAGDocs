import json

import httpx
import pytest
from pydantic import SecretStr

from ragdocs.cli import (
    EvaluationCase,
    apply_reviews,
    evaluation_fingerprint,
    evaluation_report,
    load_cases,
    run_evaluation,
    verify_live_key,
)
from ragdocs.config import Settings
from ragdocs.contracts import AnswerResult, RetrievedChunk, SourceChunk
from ragdocs.errors import ConfigurationError, CreditError, ProviderUnavailableError


def test_budget_requires_nonrenewing_limit_at_most_one():
    settings = Settings(api_key=SecretStr("fake"))
    for limit, reset, expected in [
        (1, None, None),
        (3, None, ConfigurationError),
        (None, None, ConfigurationError),
        (1, "monthly", ConfigurationError),
    ]:
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"data": {"limit": limit, "limit_reset": reset, "limit_remaining": 1}}
                )
            )
        )
        if expected:
            with pytest.raises(expected):
                verify_live_key(settings, client)
        else:
            verify_live_key(settings, client)
        client.close()


def test_budget_exhausted():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": {"limit": 1, "limit_remaining": 0}})
        )
    )
    with pytest.raises(CreditError):
        verify_live_key(Settings(api_key=SecretStr("fake")), client)
    client.close()


def test_cases_validate_before_api(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('{"schema_version": 1, "cases": []}')
    with pytest.raises(ConfigurationError):
        load_cases(path, Settings())


def test_evaluation_resume_and_manual_review(tmp_path):
    settings = Settings(project_root=tmp_path)
    cases = [
        EvaluationCase(
            id="one",
            question="O que é RAG?",
            expected_files=["guia.pdf"],
            expected_status="answered",
            rubric="Recupera antes de gerar.",
        )
    ]
    chunk = SourceChunk(
        chunk_id="chunk",
        text="Busca antes de gerar",
        document_id="doc",
        file_name="guia.pdf",
        page=1,
    )

    class Search:
        def search(self, question, k=4):
            return [RetrievedChunk(chunk=chunk, score=1)]

    calls = []

    class Service:
        def __init__(self, retriever):
            self.retriever = retriever

        def ask(self, question):
            calls.append(question)
            self.retriever.search(question)
            return AnswerResult(status="answered", answer="Busca antes de gerar.", sources=[chunk])

    path = tmp_path / "evaluation.json"
    checkpoint = run_evaluation(settings, cases, path, Search(), Service)
    resumed = run_evaluation(settings, cases, path, Search(), Service)
    assert len(calls) == 1
    assert resumed["report"]["retrieval_hits"] == 1
    assert not resumed["report"]["accepted"]
    review_file = tmp_path / "reviews.json"
    review_file.write_text(
        json.dumps(
            {
                "fingerprint": checkpoint["fingerprint"],
                "reviews": {
                    "one": {
                        "correct": True,
                        "grounded": True,
                        "notes": "Conforme ao trecho citado.",
                    }
                },
            }
        )
    )
    apply_reviews(checkpoint, review_file, cases)
    assert evaluation_report(checkpoint, cases)["accepted"]
    old = evaluation_fingerprint(settings, cases)
    fixture_dir = tmp_path / "tests" / "fixtures" / "pdfs"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "new.pdf").write_bytes(b"different bytes")
    assert evaluation_fingerprint(settings, cases) != old


def test_technical_failure_is_checkpointed_not_a_refusal(tmp_path):
    settings = Settings(project_root=tmp_path)
    cases = [
        EvaluationCase(
            id="one",
            question="RAG?",
            expected_files=[],
            expected_status="insufficient_context",
            rubric="Sem suporte.",
        )
    ]

    class Search:
        def search(self, question, k=4):
            return []

    class Failure:
        def __init__(self, retriever):
            pass

        def ask(self, question):
            raise ProviderUnavailableError("Provedor indisponível.")

    path = tmp_path / "checkpoint.json"
    with pytest.raises(ProviderUnavailableError):
        run_evaluation(settings, cases, path, Search(), Failure)
    checkpoint = json.loads(path.read_text())
    assert checkpoint["results"]["one"]["status"] == "technical_error"
    resumed = run_evaluation(settings, cases, path, Search(), Failure)
    assert resumed["report"]["completed"] == 0
    assert resumed["report"]["refusals"] == 0
