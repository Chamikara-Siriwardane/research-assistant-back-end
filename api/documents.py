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

import io
import logging
import tempfile
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlparse

import boto3
import pypdf
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Request
from google import genai
from google.genai import types as genai_types
from sqlalchemy.orm import Session

from core.config import settings
from database import SessionLocal, get_db
from models import Chat, Document
from schemas import DocumentStatusOut, DocumentUploadAcceptedOut, PresignedUrlOut
from services.vector_store import add_multimodal_pdf_pages

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


# ---------------------------------------------------------------------------
# google-genai client — lazy singleton
# ---------------------------------------------------------------------------

_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    """Return the shared google-genai client (instantiated once per process)."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=settings.gemini_api_key)
    return _genai_client


# ---------------------------------------------------------------------------
# PDF page slicing — no text extraction performed
# ---------------------------------------------------------------------------

def _slice_pdf_to_pages(pdf_bytes: bytes) -> list[bytes]:
    """Split a PDF byte stream into a list of single-page PDF byte streams.

    Uses pypdf purely for structural slicing; no text is extracted.
    """
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pages: list[bytes] = []
    for page_index in range(len(reader.pages)):
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[page_index])
        buf = io.BytesIO()
        writer.write(buf)
        pages.append(buf.getvalue())
    return pages


# ---------------------------------------------------------------------------
# Multimodal embedding — raw PDF page bytes → vector
# ---------------------------------------------------------------------------

def _embed_pdf_page(page_bytes: bytes) -> list[float]:
    """Embed a single-page PDF directly via the Gemini multimodal embedding API."""
    response = _get_genai_client().models.embed_content(
        model=settings.embedding_model,
        contents=[
            genai_types.Part.from_bytes(
                data=page_bytes,
                mime_type="application/pdf",
            )
        ],
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return list(response.embeddings[0].values)


# ---------------------------------------------------------------------------
# Background task — asynchronous RAG ingestion
# ---------------------------------------------------------------------------


def process_document_rag(document_id: int) -> None:
    """
    Download PDF from S3, slice it into individual pages (no text extraction),
    embed each page via the Gemini multimodal embedding API, and persist the
    resulting vectors with page-level metadata.

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
        pdf_bytes = temp_file_path.read_bytes()

        logger.info(f"[Doc {document_id}] Slicing PDF into individual pages...")
        page_byte_streams = _slice_pdf_to_pages(pdf_bytes)
        logger.info(f"[Doc {document_id}] PDF has {len(page_byte_streams)} page(s).")

        logger.info(f"[Doc {document_id}] Embedding pages via Gemini multimodal API...")
        embeddings: list[list[float]] = []
        for page_index, page_bytes in enumerate(page_byte_streams, start=1):
            embedding = _embed_pdf_page(page_bytes)
            embeddings.append(embedding)
            logger.debug(f"[Doc {document_id}] Embedded page {page_index}/{len(page_byte_streams)}")

        logger.info(f"[Doc {document_id}] Storing {len(embeddings)} page vector(s) in ChromaDB...")
        add_multimodal_pdf_pages(
            embeddings=embeddings,
            chat_id=doc.chat_id,
            document_id=doc.id,
        )

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
