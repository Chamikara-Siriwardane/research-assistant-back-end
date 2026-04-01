"""
api/upload.py
-------------
POST /api/upload — asynchronous document ingestion endpoint.

The route saves an uploaded PDF to a temporary file, queues a background task
that extracts text, chunks it, injects session metadata, and stores the chunks
in the shared Chroma collection.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.vector_store import add_documents

router = APIRouter()


def _ingest_pdf(temp_path: str, session_id: str, source_file: str) -> None:
    """Load, split, annotate, and persist a PDF in the background."""
    temp_file = Path(temp_path)

    try:
        loader = PyPDFLoader(str(temp_file))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks: list[Document] = splitter.split_documents(pages)

        for chunk in chunks:
            # Inject metadata before embedding so every chunk can be filtered later.
            chunk.metadata["session_id"] = session_id
            chunk.metadata["source_file"] = source_file

        add_documents(chunks)
    finally:
        # Always remove the temporary file, even if PDF parsing or embedding fails.
        temp_file.unlink(missing_ok=True)


@router.post("/upload", responses={400: {"description": "Only PDF files are supported."}})
def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    session_id: Annotated[str, Form(...)],
) -> dict:
    """Accept a PDF and queue ingestion without blocking the HTTP response."""
    original_filename = file.filename or "uploaded.pdf"
    if Path(original_filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    suffix = Path(original_filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    background_tasks.add_task(_ingest_pdf, temp_path, session_id, original_filename)

    return {"message": "Processing started.", "status": "processing"}