"""
test_rag_pipeline.py
--------------------
Standalone smoke-test for the RAG ingestion pipeline.

Calls process_document_rag(DOCUMENT_ID) directly — no mocking.
The document row (id=23) must already exist in the database with a valid S3 URL.

Usage:
    python test_rag_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so all internal imports resolve.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_rag_pipeline")

DOCUMENT_ID = 15


def test_embedding_steps(s3_url: str, file_name: str) -> None:
    """
    Smoke-test the embedding sub-steps in isolation, without touching the DB.

    Steps verified:
      1. PDF download from S3 → raw bytes
      2. Page slicing         → list of single-page PDF byte streams
      3. Single-page embed    → float vector with sensible shape and magnitude
    """
    from api.documents import _download_s3_object_to_tempfile, _slice_pdf_to_pages, _embed_pdf_page

    logger.info("-" * 60)
    logger.info("STEP 1 — Downloading PDF from S3: %s", s3_url)
    temp_path = _download_s3_object_to_tempfile(s3_url, file_name)
    try:
        pdf_bytes = temp_path.read_bytes()
        logger.info("Downloaded %d bytes.", len(pdf_bytes))
        assert len(pdf_bytes) > 0, "Downloaded PDF is empty."

        logger.info("STEP 2 — Slicing PDF into individual pages …")
        pages = _slice_pdf_to_pages(pdf_bytes)
        logger.info("Sliced into %d page(s).", len(pages))
        assert len(pages) > 0, "PDF produced zero pages after slicing."
        for i, page in enumerate(pages, start=1):
            assert isinstance(page, bytes) and len(page) > 0, (
                f"Page {i} byte stream is empty or not bytes."
            )
        logger.info("All %d page byte streams are non-empty. ✓", len(pages))

        logger.info("STEP 3 — Embedding page 1 via Gemini multimodal API …")
        vector = _embed_pdf_page(pages[0])
        dim = len(vector)
        logger.info("Embedding returned a vector of dimension %d.", dim)
        assert dim > 0, "Embedding vector is empty."
        assert all(isinstance(v, float) for v in vector), "Embedding contains non-float values."
        assert any(v != 0.0 for v in vector), "Embedding vector is all zeros — likely an API error."
        logger.info(
            "Vector looks healthy — dim=%d  min=%.6f  max=%.6f  first_5=%s ✓",
            dim,
            min(vector),
            max(vector),
            [round(v, 6) for v in vector[:5]],
        )
    finally:
        temp_path.unlink(missing_ok=True)
        logger.info("Cleaned up temporary file.")

    logger.info("Embedding step tests PASSED.")
    logger.info("-" * 60)


def main() -> None:
    # Import here so the logging config above takes effect first.
    from database import SessionLocal
    from models import Document
    from api.documents import process_document_rag

    # ── Pre-flight: confirm the document exists and show its current state ──
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == DOCUMENT_ID).first()
        if doc is None:
            logger.error("Document id=%s not found in the database. Aborting.", DOCUMENT_ID)
            sys.exit(1)

        logger.info(
            "Found document — id=%s  file_name=%s  status=%s  s3_url=%s",
            doc.id,
            doc.file_name,
            doc.status,
            doc.s3_url,
        )
        s3_url, file_name = doc.s3_url, doc.file_name
    finally:
        db.close()

    # ── Embedding sub-step tests ──
    logger.info("=" * 60)
    logger.info("Running embedding step tests …")
    logger.info("=" * 60)
    try:
        test_embedding_steps(s3_url, file_name)
    except (AssertionError, Exception) as exc:
        logger.error("Embedding step tests FAILED: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Run the full pipeline ──
    logger.info("=" * 60)
    logger.info("Starting process_document_rag(%s) …", DOCUMENT_ID)
    logger.info("=" * 60)

    process_document_rag(DOCUMENT_ID)

    # ── Post-flight: read back the updated status ──
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == DOCUMENT_ID).first()
        final_status = doc.status if doc else "unknown"
        logger.info("=" * 60)
        logger.info("Pipeline finished. Document id=%s final status: %s", DOCUMENT_ID, final_status)
        logger.info("=" * 60)
        if final_status != "ready":
            logger.error("Expected status 'ready' but got '%s'.", final_status)
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
