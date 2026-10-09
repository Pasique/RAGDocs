"""Interface local: perguntas independentes e histórico somente visual."""

import hashlib
import json
from collections.abc import Callable
from urllib.parse import urlparse

import streamlit as st

from ragdocs.config import Settings, load_settings
from ragdocs.contracts import AnswerResult, AnswerService
from ragdocs.errors import (
    AuthenticationError,
    ConfigurationError,
    CorpusError,
    CreditError,
    IndexIncompatibleError,
    IndexMissingError,
    InvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuestionError,
)

ServiceFactory = Callable[[Settings], AnswerService]


def _create_service(settings: Settings) -> AnswerService:
    # A interface pode ser testada antes de integrar índice e provedor.
    from ragdocs.service import create_service

    return create_service(settings)


def _safe_error(error: Exception) -> str:
    """Nunca apresenta mensagens brutas nem credenciais do provedor."""
    messages = {
        ConfigurationError: "Configure OPENROUTER_API_KEY no .env deste projeto.",
        CorpusError: "A coleção documental está inválida. Confira o manifesto e os arquivos.",
        IndexMissingError: "Índice ausente. Execute python -m ragdocs.cli build-index.",
        IndexIncompatibleError: (
            "O índice não corresponde à configuração ou à coleção atual. "
            "Execute python -m ragdocs.cli build-index."
        ),
        AuthenticationError: "O OpenRouter recusou a autenticação. Confira sua chave.",
        CreditError: "Crédito indisponível no OpenRouter. Verifique o limite da chave dedicada.",
        ProviderTimeoutError: "O OpenRouter demorou para responder. Tente novamente mais tarde.",
        ProviderUnavailableError: "O OpenRouter está indisponível. Tente novamente mais tarde.",
        InvalidResponseError: (
            "A resposta recebida não pôde ser validada. Nenhuma resposta foi aceita."
        ),
        QuestionError: "Digite uma pergunta entre 1 e 1.000 caracteres.",
    }
    for error_type, message in messages.items():
        if isinstance(error, error_type):
            return message
    return "Não foi possível concluir a consulta. Confira a configuração e tente novamente."


def _signature(settings: Settings) -> tuple:
    """Invalida o recurso por configuração, corpus ou alteração dos arquivos do índice."""
    paths = [settings.corpus_dir / "manifest.json"]
    if settings.corpus_dir.exists():
        paths.extend(path for path in settings.corpus_dir.rglob("*") if path.is_file())
    if settings.index_dir.exists():
        paths.extend(path for path in settings.index_dir.rglob("*") if path.is_file())
    files = []
    for path in sorted(paths):
        try:
            stat = path.stat()
            files.append((str(path), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            files.append((str(path), None, None))
    config = settings.model_dump(mode="json", exclude={"api_key"})
    key_fingerprint = hashlib.sha256(settings.api_key.get_secret_value().encode()).hexdigest()
    return json.dumps(config, sort_keys=True), key_fingerprint, tuple(files)


def _get_service(settings: Settings, factory: ServiceFactory) -> AnswerService:
    signature = _signature(settings)
    cached = st.session_state.get("_ragdocs_service")
    if cached is None or cached[0] != signature:
        service = factory(settings)
        st.session_state["_ragdocs_service"] = (signature, service)
        return service
    return cached[1]


def _is_web_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"https", "http"} and bool(parsed.netloc)


def _collection(settings: Settings) -> None:
    st.subheader("Coleção documental")
    try:
        manifest = json.loads((settings.corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        sources = manifest["sources"]
        if not isinstance(sources, list):
            raise ValueError("Manifesto inválido")
        for source in sources:
            title, url = source["title"], source["url"]
            if not isinstance(title, str) or not isinstance(url, str):
                raise ValueError("Fonte inválida")
            if _is_web_url(url):
                st.link_button(title, url, use_container_width=True)
            else:
                st.text(title)
        st.caption(f"{len(sources)} fontes fixas. As consultas não navegam na internet.")
    except (OSError, ValueError, KeyError, TypeError):
        st.caption("Coleção indisponível. Confira corpus/manifest.json.")


def _render_result(result: AnswerResult) -> None:
    if result.status == "insufficient_context":
        st.info(result.answer)
        return
    st.markdown(result.answer)
    for number, source in enumerate(result.sources, start=1):
        with st.expander(f"Fonte {number} · {source.title} · {source.chunk_id}"):
            if _is_web_url(source.url):
                st.link_button("Abrir documento original", source.url)
            st.caption(f"Revisão: {source.revision}")
            st.code(source.text, language=None, wrap_lines=True)


def main(service_factory: ServiceFactory | None = None, settings: Settings | None = None) -> None:
    """Desenha a aplicação; a factory injetável recebe a configuração imutável."""
    st.set_page_config(page_title="RAGDocs", page_icon="📚", layout="centered")
    settings = settings or load_settings()
    factory = service_factory or _create_service
    st.title("📚 RAGDocs")
    st.write(
        "Tire dúvidas sobre LangChain e FAISS com respostas em português e fontes verificáveis."
    )
    st.caption(
        "Cada pergunta é independente. O histórico abaixo serve apenas para consulta visual."
    )

    history = st.session_state.setdefault("ragdocs_history", [])
    key_ready = bool(settings.api_key.get_secret_value().strip())
    index_ready = settings.index_dir.exists() and any(
        path.is_file() for path in settings.index_dir.iterdir()
    )
    with st.sidebar:
        st.header("Sua sessão")
        st.caption("Chave OpenRouter configurada" if key_ready else "Chave OpenRouter ausente")
        st.caption("Índice local encontrado" if index_ready else "Índice local ausente")
        if st.button("Limpar histórico", use_container_width=True):
            history.clear()
        st.caption(f"{len(history)} de {settings.max_history} interações recentes")
        _collection(settings)

    if not key_ready:
        st.warning("Configure OPENROUTER_API_KEY no .env deste projeto para consultar.")
    if not index_ready:
        st.warning("Prepare o índice no terminal: python -m ragdocs.cli build-index")

    for item in history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            _render_result(item["result"])

    question = st.chat_input(
        "Pergunte sobre LangChain ou FAISS",
        max_chars=settings.max_question_chars,
        disabled=not (key_ready and index_ready),
    )
    if question is None:
        return
    question = question.strip()
    if not question or len(question) > settings.max_question_chars:
        st.error(f"Digite uma pergunta entre 1 e {settings.max_question_chars} caracteres.")
        return
    try:
        with st.spinner("Consultando a coleção…"):
            result = _get_service(settings, factory).ask(question)
    except Exception as error:
        st.error(_safe_error(error))
        return
    history.append({"question": question, "result": result})
    del history[:-settings.max_history]
    st.rerun()
