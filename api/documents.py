"""
api/documents.py
----------------
Document handling endpoints — upload, status check, and presigned URL generation.

Router prefix: /api/documents
"""

from __future__ import annotations

import time
from typing import Annotated

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from core.config import settings
from database import SessionLocal, get_db
from models import Chat, Document
from schemas import DocumentOut, DocumentStatusOut, PresignedUrlOut

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Background task — mock RAG processing
# ---------------------------------------------------------------------------


def process_document_rag(document_id: int) -> None:
    """
    Mock document processing pipeline.

    Sleeps for 10 seconds to simulate RAG ingestion, then marks the
    document as 'ready'.  Uses its own DB session because background
    tasks run outside the request lifecycle.
    """
    db = SessionLocal()
    try:
        time.sleep(10)
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "ready"
            db.commit()
    except Exception:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _get_s3_client():
    """Build a boto3 S3 client from application settings."""
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )


def _upload_to_s3(file: UploadFile, chat_id: int) -> str:
    """
    Upload a file to the configured S3 bucket.

    Returns the full S3 object URL.
    """
    s3 = _get_s3_client()
    key = f"chats/{chat_id}/{file.filename}"
    s3.upload_fileobj(
        file.file,
        settings.s3_bucket_name,
        key,
        ExtraArgs={"ContentType": file.content_type or "application/pdf"},
    )
    return f"https://{settings.s3_bucket_name}.s3.{settings.aws_region}.amazonaws.com/{key}"


def _generate_presigned_url(s3_url: str, expiration: int = 3600) -> str:
    """Generate a presigned GET URL from the stored S3 object URL."""
    s3 = _get_s3_client()
    # Extract bucket and key from the full URL
    # Format: https://<bucket>.s3.<region>.amazonaws.com/<key>
    parts = s3_url.replace("https://", "").split("/", 1)
    bucket = parts[0].split(".s3.")[0]
    key = parts[1] if len(parts) > 1 else ""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiration,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chats/{chat_id}/documents", response_model=DocumentOut, status_code=202)
def upload_document(
    chat_id: int,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    db: Session = Depends(get_db),
):
    """
    Upload a PDF to S3, create a Document record with status 'processing',
    and kick off background RAG processing.  Returns 202 Accepted immediately.
    """
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Upload to S3
    s3_url = _upload_to_s3(file, chat_id)

    # Persist document record
    document = Document(
        chat_id=chat_id,
        file_name=file.filename or "uploaded.pdf",
        s3_url=s3_url,
        status="processing",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Queue background processing
    background_tasks.add_task(process_document_rag, document.id)

    return document


@router.get("/{document_id}/status", response_model=DocumentStatusOut)
def get_document_status(document_id: int, db: Session = Depends(get_db)):
    """Return the current processing status of a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/url", response_model=PresignedUrlOut)
def get_document_url(document_id: int, db: Session = Depends(get_db)):
    """Generate and return an S3 presigned URL for downloading/previewing the PDF."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        url = _generate_presigned_url(document.s3_url)
    except ClientError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {exc}")

    return PresignedUrlOut(document_id=document.id, url=url)
