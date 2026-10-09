"""Validação e extração de PDFs recebidos, exclusivamente em memória."""

import hashlib
import json
from io import BytesIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.generic import ArrayObject

from ragdocs.config import Settings
from ragdocs.contracts import DocumentInfo, ExtractedCollection, SourceChunk, UploadedPDF
from ragdocs.errors import DocumentError


def selection_fingerprint(settings: Settings, uploads: list[UploadedPDF]) -> str:
    """Identificar seleção/configuração sem interpretar PDFs nem acessar o provedor."""
    identity = {
        "schema_version": 2,
        "files": [
            {"name": upload.name, "sha256": hashlib.sha256(upload.data).hexdigest()}
            for upload in uploads
        ],
        "processing": {
            name: getattr(settings, name)
            for name in (
                "embedding_model",
                "embedding_dimensions",
                "chunk_size",
                "chunk_overlap",
                "max_pdf_files",
                "max_pdf_bytes",
                "max_collection_bytes",
                "max_collection_pages",
                "max_extracted_chars",
                "max_page_stream_bytes",
            )
        },
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _validate_selection(settings: Settings, uploads: list[UploadedPDF]) -> None:
    if not uploads:
        raise DocumentError("Selecione pelo menos um PDF para processar.")
    if len(uploads) > settings.max_pdf_files:
        raise DocumentError(f"Selecione no máximo {settings.max_pdf_files} PDFs.")
    if any(not upload.data for upload in uploads):
        raise DocumentError("O lote contém um arquivo vazio. Remova-o e tente novamente.")
    if any(len(upload.data) > settings.max_pdf_bytes for upload in uploads):
        raise DocumentError("Um PDF excede o limite de tamanho por arquivo.")
    if sum(len(upload.data) for upload in uploads) > settings.max_collection_bytes:
        raise DocumentError("Os PDFs excedem o limite de tamanho total da coleção.")
    if settings.chunk_size <= 0 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise DocumentError("Parâmetros de divisão inválidos: confira tamanho e sobreposição.")


def _check_page_stream(page, limit: int) -> None:
    """Inspecionar tamanho descomprimido antes de executar a extração de texto."""
    contents = page.get("/Contents")
    if contents is None:
        return
    contents = contents.get_object()
    streams = contents if isinstance(contents, ArrayObject) else [contents]
    total = 0
    for stream in streams:
        total += len(stream.get_object().get_data())
        if total > limit:
            raise DocumentError("Uma página excede o limite de processamento do PDF.")


def extract_collection(settings: Settings, uploads: list[UploadedPDF]) -> ExtractedCollection:
    """Validar TODO o lote antes de embeddings; preservar arquivo e página dos trechos."""
    _validate_selection(settings, uploads)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    documents = []
    chunks = []
    seen = set()
    duplicate_count = skipped_pages = total_pages = total_chars = 0
    for upload in uploads:
        document_id = hashlib.sha256(upload.data).hexdigest()
        if document_id in seen:
            duplicate_count += 1
            continue
        seen.add(document_id)
        try:
            reader = PdfReader(BytesIO(upload.data), strict=True)
            if reader.is_encrypted:
                raise DocumentError("O lote contém um PDF protegido por senha. Remova a proteção.")
            page_count = len(reader.pages)
            total_pages += page_count
            if total_pages > settings.max_collection_pages:
                raise DocumentError("A coleção excede o limite total de páginas.")
            text_pages = 0
            for page_number, page in enumerate(reader.pages, start=1):
                _check_page_stream(page, settings.max_page_stream_bytes)
                text = (page.extract_text() or "").strip()
                if not text:
                    skipped_pages += 1
                    continue
                total_chars += len(text)
                if total_chars > settings.max_extracted_chars:
                    raise DocumentError("A coleção excede o limite total de caracteres extraídos.")
                text_pages += 1
                for position, part in enumerate(splitter.split_text(text)):
                    digest = hashlib.sha256(part.encode("utf-8")).hexdigest()[:12]
                    chunks.append(
                        SourceChunk(
                            chunk_id=f"{document_id}:{page_number:04d}:{position:04d}:{digest}",
                            text=part,
                            document_id=document_id,
                            file_name=upload.name,
                            page=page_number,
                        )
                    )
            if not text_pages:
                raise DocumentError(
                    "O lote contém um PDF sem texto extraível. PDFs escaneados exigem OCR."
                )
            documents.append(
                DocumentInfo(
                    document_id=document_id,
                    file_name=upload.name,
                    page_count=page_count,
                    text_pages=text_pages,
                )
            )
        except DocumentError:
            raise
        except Exception:
            # Erros do parser podem conter dados recebidos; nunca expô-los na interface.
            raise DocumentError("O lote contém um PDF inválido ou que não pôde ser lido.") from None
    if not chunks:
        raise DocumentError("Os PDFs não produziram trechos para consulta.")
    return ExtractedCollection(
        fingerprint=selection_fingerprint(settings, uploads),
        documents=documents,
        chunks=chunks,
        skipped_pages=skipped_pages,
        duplicate_count=duplicate_count,
    )
