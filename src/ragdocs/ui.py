"""PDFs e índice isolados na sessão; perguntas independentes e fontes por página."""

import hashlib
import json
from collections.abc import Callable

import streamlit as st

from ragdocs.config import Settings, load_settings
from ragdocs.contracts import (
    AnswerResult,
    AnswerService,
    PreparedCollection,
    Retriever,
    UploadedPDF,
)
from ragdocs.errors import (
    AuthenticationError,
    ConfigurationError,
    CreditError,
    DocumentError,
    InvalidResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuestionError,
)

ServiceFactory = Callable[[Settings, Retriever], AnswerService]
Processor = Callable[[Settings, list[UploadedPDF]], PreparedCollection]
UploadProvider = Callable[[str], list[UploadedPDF]]


def _create_service(settings: Settings, retriever: Retriever) -> AnswerService:
    from ragdocs.service import create_service

    return create_service(settings, retriever)


def _process_pdfs(settings: Settings, uploads: list[UploadedPDF]) -> PreparedCollection:
    from ragdocs.index import process_pdfs

    return process_pdfs(settings, uploads)


def _selection_fingerprint(settings: Settings, uploads: list[UploadedPDF]) -> str:
    from ragdocs.ingestion import selection_fingerprint

    return selection_fingerprint(settings, uploads)


def _signature(settings: Settings, uploads: list[UploadedPDF]) -> str:
    config = json.dumps(settings.model_dump(mode="json", exclude={"api_key"}), sort_keys=True)
    key_hash = hashlib.sha256(settings.api_key.get_secret_value().encode()).hexdigest()
    content = _selection_fingerprint(settings, uploads)
    return hashlib.sha256(f"{config}:{key_hash}:{content}".encode()).hexdigest()


def _safe_error(error: Exception) -> str:
    """Mensagens controladas: nunca apresenta erro bruto do provedor ou conteúdo do PDF."""
    if isinstance(error, DocumentError):
        document_messages = {
            "empty_selection": "Selecione pelo menos um PDF para processar.",
            "file_count": "A seleção excede a quantidade máxima de PDFs. Remova alguns arquivos.",
            "empty_file": "Um dos arquivos está vazio. Remova-o e tente novamente.",
            "file_size": "Um PDF excede o limite de tamanho por arquivo indicado acima.",
            "collection_size": "Os PDFs excedem o limite de tamanho total indicado acima.",
            "page_limit": "A coleção excede o limite de páginas indicado acima. Use menos páginas.",
            "character_limit": "A coleção excede o limite de caracteres extraídos indicado acima.",
            "page_stream": "Uma página é complexa demais para o limite de processamento do MVP.",
            "split_settings": "A configuração de divisão dos documentos precisa ser corrigida.",
            "password_required": (
                "Um dos PDFs exige senha para abrir. O MVP não aceita senhas; "
                "envie uma cópia que abra sem pedir senha."
            ),
            "no_text": (
                "Um dos PDFs não tem texto extraível. Documentos digitalizados como imagens "
                "precisam de OCR, que não está disponível neste MVP."
            ),
            "dependency": (
                "Falta o suporte de criptografia para ler esse PDF. "
                "Atualize o ambiente seguindo a instalação do README."
            ),
        }
        return document_messages.get(
            error.code,
            "Não foi possível ler um dos PDFs. Use arquivos válidos, sem senha de abertura "
            "e com texto extraível. Remova o arquivo inválido e tente novamente.",
        )
    messages = {
        ConfigurationError: "Configure OPENROUTER_API_KEY no .env deste projeto.",
        AuthenticationError: "O OpenRouter recusou a autenticação. Confira sua chave.",
        CreditError: "Crédito indisponível no OpenRouter. Verifique o limite da chave dedicada.",
        ProviderTimeoutError: "O OpenRouter demorou para responder. Tente novamente mais tarde.",
        ProviderUnavailableError: "O OpenRouter está indisponível. Tente novamente mais tarde.",
        InvalidResponseError: (
            "A resposta recebida não pôde ser validada. Nenhuma resposta foi aceita."
        ),
        QuestionError: "Digite uma pergunta dentro do limite indicado.",
    }
    for error_type, message in messages.items():
        if isinstance(error, error_type):
            return message
    return "Não foi possível concluir a operação. Confira a configuração e tente novamente."


def _invalidate_collection() -> None:
    for key in ("_ragdocs_prepared", "_ragdocs_service"):
        st.session_state.pop(key, None)
    st.session_state["ragdocs_history"] = []


def _render_result(result: AnswerResult) -> None:
    if result.status == "insufficient_context":
        st.info(result.answer)
        return
    st.markdown(result.answer)
    for number, source in enumerate(result.sources, start=1):
        with st.expander(f"Fonte {number} · {source.file_name} — página {source.page}"):
            st.caption(f"Trecho: {source.chunk_id}")
            st.code(source.text, language=None, wrap_lines=True)


def _uploaded_files(key: str, settings: Settings) -> list[UploadedPDF]:
    files = st.file_uploader(
        "Selecione seus PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=key,
        max_upload_size=settings.max_pdf_bytes // (1024 * 1024),
    )
    return [UploadedPDF(name=file.name, data=file.getvalue()) for file in files]


def main(
    service_factory: ServiceFactory | None = None,
    settings: Settings | None = None,
    processor: Processor | None = None,
    upload_provider: UploadProvider | None = None,
) -> None:
    """Injeções permitem verificar a sessão sem upload nativo nem chamadas pagas."""
    st.set_page_config(page_title="RAGDocs", page_icon="📚", layout="centered")
    settings = settings or load_settings()
    factory = service_factory or _create_service
    process = processor or _process_pdfs
    st.title("📚 RAGDocs")
    st.write("Envie PDFs e consulte seu conteúdo com respostas em português e fontes verificáveis.")
    st.caption(
        "Trechos dos PDFs serão enviados ao OpenRouter para embeddings e respostas. "
        "Os documentos e o índice ficam somente na memória desta sessão."
    )
    st.caption("Cada pergunta é independente. O histórico serve apenas para consulta visual.")
    st.caption(
        f"Até {settings.max_pdf_files} PDFs, "
        f"{settings.max_pdf_bytes // (1024 * 1024)} MiB por arquivo e "
        f"{settings.max_collection_bytes // (1024 * 1024)} MiB no conjunto; "
        f"{settings.max_collection_pages} páginas e "
        f"{settings.max_extracted_chars:,} caracteres extraídos. "
        "PDFs com texto extraível que abram sem pedir senha. OCR não está disponível."
    )

    generation = st.session_state.setdefault("_ragdocs_upload_generation", 0)
    uploader_key = f"ragdocs_uploads_{generation}"
    uploads = (
        upload_provider(uploader_key)
        if upload_provider is not None
        else _uploaded_files(uploader_key, settings)
    )
    signature = _signature(settings, uploads)
    if st.session_state.get("_ragdocs_selection") != signature:
        _invalidate_collection()
        st.session_state["_ragdocs_selection"] = signature
    st.session_state["_ragdocs_uploads"] = uploads
    history = st.session_state.setdefault("ragdocs_history", [])
    key_ready = bool(settings.api_key.get_secret_value().strip())

    with st.sidebar:
        st.header("Sua sessão")
        st.caption("Chave OpenRouter configurada" if key_ready else "Chave OpenRouter ausente")
        if st.button("Limpar documentos", use_container_width=True):
            _invalidate_collection()
            st.session_state.pop("_ragdocs_selection", None)
            st.session_state.pop(uploader_key, None)
            st.session_state["_ragdocs_uploads"] = []
            st.session_state["_ragdocs_upload_generation"] = generation + 1
            st.rerun()
        if st.button("Limpar histórico", use_container_width=True):
            history.clear()
        st.caption(f"{len(history)} de {settings.max_history} interações recentes")

    if not key_ready:
        st.warning("Configure OPENROUTER_API_KEY no .env deste projeto para processar e consultar.")
    if st.button("Processar documentos", disabled=not (uploads and key_ready)):
        _invalidate_collection()
        history = st.session_state["ragdocs_history"]
        try:
            with st.spinner("Processando documentos…"):
                prepared = process(settings, uploads)
                service = factory(settings, prepared.retriever)
            st.session_state["_ragdocs_prepared"] = prepared
            st.session_state["_ragdocs_service"] = service
        except Exception as error:
            st.error(_safe_error(error))

    prepared = st.session_state.get("_ragdocs_prepared")
    if prepared is None:
        st.info("Selecione os PDFs e clique em Processar documentos para começar.")
    else:
        collection = prepared.collection
        st.success(f"{len(collection.documents)} documento(s) pronto(s) para consulta.")
        with st.sidebar:
            st.subheader("Documentos processados")
            for document in collection.documents:
                st.text(document.file_name)
                st.caption(f"{document.text_pages} de {document.page_count} páginas com texto")
        if collection.skipped_pages:
            st.warning(f"{collection.skipped_pages} página(s) sem texto ficaram fora da consulta.")
        if collection.duplicate_count:
            st.caption(f"{collection.duplicate_count} arquivo(s) idêntico(s) ignorado(s).")

    for item in history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            _render_result(item["result"])

    question = st.chat_input(
        "Pergunte sobre seus documentos",
        max_chars=settings.max_question_chars,
        disabled=not (key_ready and prepared is not None),
    )
    if question is None or not key_ready or prepared is None:
        return
    question = question.strip()
    if not question or len(question) > settings.max_question_chars:
        st.error(f"Digite uma pergunta entre 1 e {settings.max_question_chars} caracteres.")
        return
    try:
        with st.spinner("Consultando os documentos…"):
            result = st.session_state["_ragdocs_service"].ask(question)
    except Exception as error:
        st.error(_safe_error(error))
        return
    history.append({"question": question, "result": result})
    del history[: -settings.max_history]
    st.rerun()
