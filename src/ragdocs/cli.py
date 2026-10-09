"""Operação explícita do índice e avaliação real retomável."""

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

from ragdocs.config import Settings, load_settings
from ragdocs.contracts import AnswerResult, AnswerService, RetrievedChunk, Retriever
from ragdocs.errors import ConfigurationError, CreditError, RagDocsError


class EvaluationCase(BaseModel):
    id: str
    question: str
    expected_document_ids: list[str]
    expected_status: Literal["answered", "insufficient_context"]
    rubric: str


class RecordingRetriever:
    def __init__(self, delegate: Retriever):
        self.delegate = delegate
        self.last_results: list[RetrievedChunk] = []

    def search(self, question: str, k: int = 4) -> list[RetrievedChunk]:
        self.last_results = self.delegate.search(question, k)
        return self.last_results


def verify_live_key(settings: Settings, client: httpx.Client | None = None) -> None:
    """Verifica o teto dedicado antes de gastar; não revela metadados da chave."""
    own_client = client is None
    client = client or httpx.Client(timeout=settings.timeout_seconds)
    try:
        response = client.get(
            settings.base_url + "/key",
            headers={"Authorization": f"Bearer {settings.require_api_key()}"},
        )
        if response.status_code != 200:
            raise ConfigurationError("Não foi possível verificar o limite da chave OpenRouter.")
        data = response.json()["data"]
        limit = data.get("limit")
        remaining = data.get("limit_remaining")
        if (
            not isinstance(limit, (int, float))
            or isinstance(limit, bool)
            or not 0 < limit <= 1
            or data.get("limit_reset") is not None
        ):
            raise ConfigurationError(
                "Use uma chave dedicada com limite total de até US$ 1, sem renovação."
            )
        if not isinstance(remaining, (int, float)) or remaining <= 0:
            raise CreditError("O limite de crédito da chave foi atingido.")
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise ConfigurationError("Falha ao verificar o limite da chave OpenRouter.") from exc
    finally:
        if own_client:
            client.close()


def load_cases(path: Path, settings: Settings) -> list[EvaluationCase]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        if body["schema_version"] != 1:
            raise ValueError("schema")
        cases = [EvaluationCase.model_validate(item) for item in body["cases"]]
        if not cases or len({case.id for case in cases}) != len(cases):
            raise ValueError("ids")
        for case in cases:
            if not case.question.strip() or len(case.question) > settings.max_question_chars:
                raise ValueError("question")
            if bool(case.expected_document_ids) != (case.expected_status == "answered"):
                raise ValueError("expectation")
        return cases
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        raise ConfigurationError("Arquivo de perguntas de avaliação inválido.") from exc


def evaluation_fingerprint(settings: Settings, cases: list[EvaluationCase]) -> str:
    digest = hashlib.sha256()
    config = {
        "chat": settings.chat_model,
        "embedding": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
        "size": settings.chunk_size,
        "overlap": settings.chunk_overlap,
        "k": settings.top_k,
        "output_tokens": settings.max_output_tokens,
        "cases": [case.model_dump() for case in cases],
    }
    digest.update(json.dumps(config, sort_keys=True, ensure_ascii=False).encode())
    for path in sorted(settings.corpus_dir.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(settings.corpus_dir)).encode())
            digest.update(path.read_bytes())
    for name in ["product_prompt.py", "service.py", "providers.py", "contracts.py", "cli.py"]:
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_checkpoint(path: Path, fingerprint: str) -> dict:
    if not path.exists():
        return {"schema_version": 1, "fingerprint": fingerprint, "results": {}}
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if (
            checkpoint["schema_version"] != 1
            or checkpoint["fingerprint"] != fingerprint
            or not isinstance(checkpoint["results"], dict)
        ):
            raise ValueError("checkpoint")
        return checkpoint
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigurationError("Checkpoint da avaliação inválido ou incompatível.") from exc


def evaluation_report(checkpoint: dict, cases: list[EvaluationCase]) -> dict:
    results = checkpoint["results"]
    supported = [case for case in cases if case.expected_status == "answered"]
    unsupported = [case for case in cases if case.expected_status == "insufficient_context"]
    completed = [result for result in results.values() if result.get("status") != "technical_error"]
    retrieval_hits = sum(
        bool(
            set(case.expected_document_ids) & set(results.get(case.id, {}).get("retrieved_ids", []))
        )
        for case in supported
    )
    refusals = sum(
        results.get(case.id, {}).get("status") == "insufficient_context" for case in unsupported
    )
    reviews = [results.get(case.id, {}).get("manual_review", {}) for case in supported]
    correct = sum(
        review.get("correct") is True and review.get("grounded") is True for review in reviews
    )
    citations_valid = all(result.get("citations_valid") is True for result in completed)
    required_hits = (8 * len(supported) + 9) // 10
    accepted = (
        len(completed) == len(cases)
        and retrieval_hits >= required_hits
        and correct >= required_hits
        and refusals == len(unsupported)
        and citations_valid
    )
    return {
        "completed": len(completed),
        "total": len(cases),
        "retrieval_hits": retrieval_hits,
        "supported_total": len(supported),
        "refusals": refusals,
        "unsupported_total": len(unsupported),
        "correct_and_grounded": correct,
        "manual_review_count": sum(bool(review) for review in reviews),
        "citations_valid": citations_valid if completed else False,
        "accepted": accepted,
    }


def run_evaluation(
    settings: Settings,
    cases: list[EvaluationCase],
    checkpoint_path: Path,
    retriever: Retriever,
    service_factory: Callable[[Retriever], AnswerService],
    retry_failed: bool = False,
) -> dict:
    checkpoint = _load_checkpoint(checkpoint_path, evaluation_fingerprint(settings, cases))
    recording = RecordingRetriever(retriever)
    service = service_factory(recording)
    for case in cases:
        existing = checkpoint["results"].get(case.id)
        if existing and not (retry_failed and existing.get("status") == "technical_error"):
            continue
        recording.last_results = []
        result = {"question": case.question, "rubric": case.rubric}
        try:
            answer: AnswerResult = service.ask(case.question)
            retrieved = recording.last_results
            valid_ids = {item.chunk.chunk_id for item in retrieved}
            result.update(answer.model_dump())
            result["retrieved_ids"] = list(
                dict.fromkeys(item.chunk.document_id for item in retrieved)
            )
            result["retrieved_chunks"] = [item.model_dump() for item in retrieved]
            result["citations_valid"] = all(
                source.chunk_id in valid_ids for source in answer.sources
            ) and (bool(answer.sources) if answer.status == "answered" else not answer.sources)
        except RagDocsError as exc:
            result.update(status="technical_error", error=str(exc))
            checkpoint["results"][case.id] = result
            _save(checkpoint_path, checkpoint)
            raise
        checkpoint["results"][case.id] = result
        _save(checkpoint_path, checkpoint)
        print(f"{case.id}: {answer.status}")
    checkpoint["report"] = evaluation_report(checkpoint, cases)
    _save(checkpoint_path, checkpoint)
    return checkpoint


def apply_reviews(checkpoint: dict, review_path: Path, cases: list[EvaluationCase]) -> None:
    """Importa revisão humana explícita; não contrata um modelo avaliador."""
    try:
        reviews = json.loads(review_path.read_text(encoding="utf-8"))
        if reviews["fingerprint"] != checkpoint["fingerprint"]:
            raise ValueError("fingerprint")
        known_ids = {case.id for case in cases if case.expected_status == "answered"}
        for case_id, review in reviews["reviews"].items():
            if (
                case_id not in known_ids
                or case_id not in checkpoint["results"]
                or not isinstance(review.get("correct"), bool)
                or not isinstance(review.get("grounded"), bool)
                or not isinstance(review.get("notes"), str)
                or not review["notes"].strip()
            ):
                raise ValueError("review")
            checkpoint["results"][case_id]["manual_review"] = review
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigurationError(
            "Revisão manual inválida; confira fingerprint e rubricas."
        ) from exc


def _build(settings: Settings) -> None:
    from ragdocs.index import build_index, load_index

    try:
        load_index(settings)
    except RagDocsError:
        verify_live_key(settings)
        build_index(settings)
        print("Índice construído em runtime/index.")
    else:
        print("Índice compatível reaproveitado; nenhum documento foi reindexado.")


def _evaluate(settings: Settings, args: argparse.Namespace) -> None:
    from ragdocs.index import load_index
    from ragdocs.service import RagAnswerService

    cases = load_cases(settings.project_root / "tests" / "fixtures" / "evaluation.json", settings)
    fingerprint = evaluation_fingerprint(settings, cases)
    path = settings.evaluation_dir / f"{fingerprint}.json"
    checkpoint = _load_checkpoint(path, fingerprint)
    if args.review_file:
        apply_reviews(checkpoint, Path(args.review_file), cases)
        checkpoint["report"] = evaluation_report(checkpoint, cases)
        _save(path, checkpoint)
    elif args.live:
        pending = [
            case
            for case in cases
            if case.id not in checkpoint["results"]
            or (
                args.retry_failed
                and checkpoint["results"][case.id].get("status") == "technical_error"
            )
        ]
        if pending:
            verify_live_key(settings)
            checkpoint = run_evaluation(
                settings,
                cases,
                path,
                load_index(settings),
                lambda retriever: RagAnswerService(settings, retriever),
                args.retry_failed,
            )
    elif not args.report:
        raise ConfigurationError("Avaliação paga exige --live; use --report para ler resultados.")
    print(json.dumps(evaluation_report(checkpoint, cases), ensure_ascii=False, indent=2))
    print(f"Resultados locais: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operação local do RAGDocs")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-index", help="Construir ou reaproveitar índice compatível")
    evaluate = commands.add_parser("evaluate", help="Avaliar e revisar a coleção fixa")
    mode = evaluate.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Autorizar chamadas reais pagas")
    mode.add_argument("--report", action="store_true", help="Exibir métricas sem chamadas")
    mode.add_argument("--review-file", help="Importar JSON de revisão humana sem chamadas")
    evaluate.add_argument("--retry-failed", action="store_true", help="Repetir falhas técnicas")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        if args.command == "build-index":
            _build(settings)
        else:
            if args.retry_failed and not args.live:
                raise ConfigurationError("--retry-failed exige --live.")
            _evaluate(settings, args)
        return 0
    except RagDocsError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
