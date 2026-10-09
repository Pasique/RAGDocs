from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ragdocs.config import Settings
from ragdocs.contracts import UploadedPDF
from ragdocs.errors import DocumentError
from ragdocs.ingestion import extract_collection, selection_fingerprint


def make_pdf(texts, password=None):
    """PDFs sintéticos determinísticos criados em memória, sem dependências auxiliares."""
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(password)
    result = BytesIO()
    writer.write(result)
    return result.getvalue()


def upload(name="documento.pdf", texts=None):
    return UploadedPDF(name=name, data=make_pdf(texts or ["Texto de teste."]))


def test_split_preserves_pages_origin_and_deterministic_identity():
    settings = Settings(chunk_size=100, chunk_overlap=20)
    uploads = [upload("manual.pdf", ["Abacate. " * 30, None, "Batata. " * 30])]
    collection = extract_collection(settings, uploads)
    assert collection == extract_collection(settings, uploads)
    assert len(collection.chunks) > 2
    assert {chunk.page for chunk in collection.chunks} == {1, 3}
    assert all(chunk.file_name == "manual.pdf" for chunk in collection.chunks)
    assert all(0 < len(chunk.text) <= 100 for chunk in collection.chunks)
    assert len({chunk.chunk_id for chunk in collection.chunks}) == len(collection.chunks)
    assert collection.skipped_pages == 1
    assert collection.documents[0].page_count == 3
    assert collection.documents[0].text_pages == 2


def test_duplicates_are_counted_and_use_first_name():
    first = upload("original.pdf")
    second = UploadedPDF(name="copia.pdf", data=first.data)
    collection = extract_collection(Settings(), [first, second])
    assert collection.duplicate_count == 1
    assert len(collection.documents) == 1
    assert {chunk.file_name for chunk in collection.chunks} == {"original.pdf"}


def test_same_names_different_contents_remain_separate():
    collection = extract_collection(Settings(), [upload(texts=["Um"]), upload(texts=["Dois"])])
    assert len(collection.documents) == 2
    assert len({chunk.document_id for chunk in collection.chunks}) == 2
    assert len({chunk.chunk_id for chunk in collection.chunks}) == 2


def test_fingerprint_detects_content_name_and_processing_changes_without_parse():
    settings = Settings()
    uploads = [UploadedPDF(name="invalid.pdf", data=b"invalid pdf")]
    identity = selection_fingerprint(settings, uploads)
    assert identity == selection_fingerprint(settings, uploads)
    for changed in [
        [UploadedPDF(name="outro.pdf", data=b"invalid pdf")],
        [UploadedPDF(name="invalid.pdf", data=b"changed")],
        [],
    ]:
        assert selection_fingerprint(settings, changed) != identity
    for update in [{"chunk_size": 100}, {"embedding_model": "outro/modelo"}]:
        assert selection_fingerprint(settings.model_copy(update=update), uploads) != identity


@pytest.mark.parametrize("data", [b"", b"not a PDF", b"%PDF-1.5\ncorrompido"])
def test_invalid_pdf_has_safe_error(data):
    with pytest.raises(DocumentError) as error:
        extract_collection(Settings(), [UploadedPDF(name="private.pdf", data=data)])
    assert "private.pdf" not in str(error.value)
    assert "corrompido" not in str(error.value)


@pytest.mark.parametrize("texts", [[None], [""], ["  \t"], []])
def test_no_text_pdf_rejects_entire_batch(texts):
    with pytest.raises(DocumentError, match="sem texto"):
        extract_collection(
            Settings(), [upload(), UploadedPDF(name="vazio.pdf", data=make_pdf(texts))]
        )


def test_encrypted_pdf_rejected_even_if_password_empty():
    for password in ["secret", ""]:
        with pytest.raises(DocumentError, match="senha"):
            extract_collection(
                Settings(),
                [
                    UploadedPDF(name="segredo.pdf", data=make_pdf(["Segredo"], password)),
                ],
            )


@pytest.mark.parametrize(
    "update,uploads,match",
    [
        ({}, [], "pelo menos"),
        ({"max_pdf_files": 1}, [upload(), upload("outro.pdf")], "máximo"),
        ({"max_pdf_bytes": 1}, [upload()], "por arquivo"),
        ({"max_collection_bytes": 1}, [upload()], "tamanho total"),
        ({"max_collection_pages": 1}, [upload(texts=["a", "b"])], "páginas"),
        ({"max_extracted_chars": 1}, [upload(texts=["texto longo"])], "caracteres"),
        ({"max_page_stream_bytes": 1}, [upload()], "página"),
        ({"chunk_overlap": 1200}, [upload()], "divisão"),
        ({"chunk_size": 0}, [upload()], "divisão"),
    ],
)
def test_collection_limits(update, uploads, match):
    with pytest.raises(DocumentError, match=match):
        extract_collection(Settings().model_copy(update=update), uploads)


def test_total_page_and_text_limits_apply_across_documents():
    uploads = [upload(texts=["abcdef"]), upload("outro.pdf", ["ghijkl"])]
    for limit in [{"max_collection_pages": 1}, {"max_extracted_chars": 10}]:
        with pytest.raises(DocumentError):
            extract_collection(Settings().model_copy(update=limit), uploads)


def test_limits_do_not_truncate_and_accept_exact_boundary():
    item = upload(texts=["abcdef"])
    settings = Settings(
        max_extracted_chars=6,
        max_collection_pages=1,
        max_pdf_bytes=len(item.data),
        max_collection_bytes=len(item.data),
    )
    assert extract_collection(settings, [item]).chunks[0].text == "abcdef"


def test_extraction_never_writes_files(tmp_path):
    collection = extract_collection(Settings(project_root=tmp_path), [upload()])
    assert collection.chunks
    assert list(tmp_path.iterdir()) == []
