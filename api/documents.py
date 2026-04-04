"""
api/documents.py
----------------
Document handling endpoints for upload, status polling, and presigned URL access.

Routes exposed under /api:
- POST /chats/{chat_id}/documents
- GET /documents/{document_id}/status
- GET /documents/{document_id}/url
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Request
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.renderers.markdown import MarkdownOutput
from sqlalchemy.orm import Session

from core.config import settings
from database import SessionLocal, get_db
from models import Chat, Document
from schemas import DocumentStatusOut, DocumentUploadAcceptedOut, PresignedUrlOut
from services.vector_store import add_document_chunks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


# ---------------------------------------------------------------------------
# marker-pdf — lazy model singleton and PDF-to-Markdown helpers
# ---------------------------------------------------------------------------

_marker_model_dict: dict | None = None
_marker_model_lock = threading.Lock()

# Headers marker uses to separate pages when paginate_output=True
_PAGE_SEPARATOR = "-" * 48

# Markdown heading levels forwarded to MarkdownHeaderTextSplitter
_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Soft ceiling for secondary character-level splitting
_MAX_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 100


def _get_marker_models() -> dict:
    """Load and cache the marker-pdf deep learning models (called once per process)."""
    global _marker_model_dict
    if _marker_model_dict is None:
        with _marker_model_lock:
            if _marker_model_dict is None:
                logger.info("Loading marker-pdf models (first call — may take a moment)...")
                _marker_model_dict = create_model_dict()
                logger.info("marker-pdf models loaded successfully.")
    return _marker_model_dict


def _convert_pdf_to_markdown(pdf_path: Path) -> MarkdownOutput:
    """Run marker-pdf on *pdf_path* and return a MarkdownOutput with page-separated Markdown."""
    converter = PdfConverter(
        artifact_dict=_get_marker_models(),
        config={"paginate_output": True, "extract_images": False},
    )
    return converter(str(pdf_path))


def _chunk_markdown(rendered: MarkdownOutput) -> list[LangChainDocument]:
    """Split marker Markdown into structurally-aware LangChain Documents.

    Strategy:
    1. Split the full markdown on marker's page-separator to recover per-page text
       and annotate each chunk with ``page`` metadata.
    2. Within each page apply ``MarkdownHeaderTextSplitter`` so chunks respect
       section boundaries and carry header-breadcrumb metadata (h1 / h2 / h3).
    3. Apply ``RecursiveCharacterTextSplitter`` as a secondary pass so oversized
       sections are further reduced without losing the header metadata.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_MAX_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )

    all_chunks: list[LangChainDocument] = []
    pages = rendered.markdown.split(_PAGE_SEPARATOR)

    for page_num, page_md in enumerate(pages, start=1):
        page_md = page_md.strip()
        if not page_md:
            continue

        header_chunks = md_splitter.split_text(page_md)
        sized_chunks = char_splitter.split_documents(header_chunks)

        for chunk in sized_chunks:
            chunk.metadata["page"] = page_num
            all_chunks.append(chunk)

    return all_chunks


# ---------------------------------------------------------------------------
# Background task — asynchronous RAG ingestion
# ---------------------------------------------------------------------------


def process_document_rag(document_id: int) -> None:
    """
    Download PDF from S3, chunk it, generate embeddings, and update status.

    Uses its own DB session because background tasks run outside the request
    lifecycle.
    """
    logger.info(f"Starting background RAG ingestion for document_id={document_id}")
    db = SessionLocal()
    temp_file_path: Path | None = None

    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            logger.warning(f"Document ID {document_id} not found in database. Aborting.")
            return

        logger.info(f"[Doc {document_id}] Downloading from S3: {doc.s3_url}")
        temp_file_path = _download_s3_object_to_tempfile(doc.s3_url, doc.file_name)

        logger.info(f"[Doc {document_id}] Converting PDF to Markdown with marker-pdf...")
        rendered = _convert_pdf_to_markdown(temp_file_path)
        logger.info(f"[Doc {document_id}] Markdown extraction complete.")

        logger.info(f"[Doc {document_id}] Splitting Markdown into structured chunks...")
        chunks = _chunk_markdown(rendered)
        logger.info(f"[Doc {document_id}] Created {len(chunks)} chunks.")

        logger.info(f"[Doc {document_id}] Embedding and storing chunks in vector DB...")
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [chunk.metadata for chunk in chunks]
        add_document_chunks(chunk_texts, chat_id=doc.chat_id, document_id=doc.id, extra_metadatas=chunk_metadatas)

        doc.status = "ready"
        db.commit()
        logger.info(f"[Doc {document_id}] Ingestion complete. Status marked as 'ready'.")
    except Exception as e:
        logger.error(f"[Doc {document_id}] Failed during RAG ingestion: {e}", exc_info=True)
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
    finally:
        if temp_file_path is not None:
            temp_file_path.unlink(missing_ok=True)
            logger.info(f"[Doc {document_id}] Cleaned up temporary file.")
        db.close()


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _get_s3_client():
    """Build a boto3 S3 client from application settings."""
    client_kwargs = {"region_name": settings.aws_region}

    if settings.aws_access_key_id and settings.aws_secret_access_key:
        client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

    return boto3.client("s3", **client_kwargs)


def _split_s3_url(s3_url: str) -> tuple[str, str]:
    """Extract bucket and key from an S3 URL in virtual-host format."""
    parsed = urlparse(s3_url)
    bucket = parsed.netloc.split(".s3.")[0]
    key = unquote(parsed.path.lstrip("/"))

    if not bucket or not key:
        raise ValueError("Invalid S3 URL")

    return bucket, key


def _upload_to_s3(file: UploadFile, chat_id: int, file_name: str) -> str:
    """
    Upload a file to the configured S3 bucket.

    Returns the full S3 object URL.
    """
    s3 = _get_s3_client()
    key = f"chats/{chat_id}/{file_name}"
    s3.upload_fileobj(
        file.file,
        settings.s3_bucket_name,
        key,
        ExtraArgs={"ContentType": file.content_type or "application/pdf"},
    )
    return f"https://{settings.s3_bucket_name}.s3.{settings.aws_region}.amazonaws.com/{key}"


def _download_s3_object_to_tempfile(s3_url: str, source_file_name: str) -> Path:
    """Download an S3 object to a temporary local file and return its path."""
    bucket, key = _split_s3_url(s3_url)
    suffix = Path(source_file_name).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        _get_s3_client().download_fileobj(bucket, key, temp_file)
        return Path(temp_file.name)


def _generate_presigned_url(s3_url: str, expiration: int = 3600) -> str:
    """Generate a presigned GET URL from the stored S3 object URL."""
    s3 = _get_s3_client()
    bucket, key = _split_s3_url(s3_url)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiration,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/chats/{chat_id}/documents",
    response_model=DocumentUploadAcceptedOut,
    status_code=202,
    responses={
        400: {"description": "Only PDF files are supported."},
        404: {"description": "Chat not found."},
        500: {"description": "S3 upload failed."},
    },
)
def upload_document(
    request: Request,
    chat_id: int,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_db)],
):
    """
    Upload a PDF to S3, create a Document row with status 'processing',
    and trigger asynchronous RAG processing.
    """
    logger.info(
        "[POST /api/chats/%s/documents] Request received from client=%s filename=%s content_type=%s",
        chat_id,
        request.client.host if request.client else "unknown",
        file.filename,
        file.content_type,
    )

    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        logger.warning("[POST /api/chats/%s/documents] Chat not found", chat_id)
        raise HTTPException(status_code=404, detail="Chat not found")

    file_name = file.filename or "uploaded.pdf"
    if Path(file_name).suffix.lower() != ".pdf":
        logger.warning(
            "[POST /api/chats/%s/documents] Rejected non-PDF file: %s",
            chat_id,
            file_name,
        )
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        logger.info("[POST /api/chats/%s/documents] Uploading file to S3", chat_id)
        s3_url = _upload_to_s3(file, chat_id, file_name)
        logger.info("[POST /api/chats/%s/documents] S3 upload complete: %s", chat_id, s3_url)
    except (BotoCoreError, ClientError) as exc:
        logger.exception("[POST /api/chats/%s/documents] S3 upload failed", chat_id)
        raise HTTPException(status_code=500, detail=f"Failed to upload document to S3: {exc}")

    document = Document(
        chat_id=chat_id,
        file_name=file_name,
        s3_url=s3_url,
        status="processing",
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info(
        "[POST /api/chats/%s/documents] Document created document_id=%s status=%s",
        chat_id,
        document.id,
        document.status,
    )

    background_tasks.add_task(process_document_rag, document.id)
    logger.info(
        "[POST /api/chats/%s/documents] Background ingestion queued for document_id=%s",
        chat_id,
        document.id,
    )

    return DocumentUploadAcceptedOut(
        document_id=document.id,
        file_name=document.file_name,
        status=document.status,
    )


@router.get(
    "/documents/{document_id}/status",
    response_model=DocumentStatusOut,
    responses={404: {"description": "Document not found."}},
)
def get_document_status(document_id: int, db: Annotated[Session, Depends(get_db)]):
    """Return the current processing status of a document."""
    logger.info("[GET /api/documents/%s/status] Status request received", document_id)
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        logger.warning("[GET /api/documents/%s/status] Document not found", document_id)
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info(
        "[GET /api/documents/%s/status] Returning status=%s",
        document_id,
        document.status,
    )
    return DocumentStatusOut(document_id=document.id, status=document.status)


@router.get(
    "/documents/{document_id}/url",
    response_model=PresignedUrlOut,
    responses={
        404: {"description": "Document not found."},
        500: {"description": "Failed to generate presigned URL."},
    },
)
def get_document_url(document_id: int, db: Annotated[Session, Depends(get_db)]):
    """Generate and return an S3 presigned URL for downloading/previewing the PDF."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        url = _generate_presigned_url(document.s3_url)
    except ClientError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {exc}")

    return PresignedUrlOut(document_id=document.id, url=url)
