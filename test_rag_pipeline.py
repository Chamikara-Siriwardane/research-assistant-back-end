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
import os
import sys
from pathlib import Path

# Use CPU — the venv has a CPU-only PyTorch build.
# To use GPU, reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121
os.environ.setdefault("TORCH_DEVICE", "cuda")

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


def main() -> None:
    # Import here so the logging config above takes effect first.
    from database import SessionLocal
    from models import Document
    from api.documents import process_document_rag

    import torch
    logger.info("Torch device: %s (CUDA available: %s)", os.environ.get("TORCH_DEVICE"), torch.cuda.is_available())

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
    finally:
        db.close()

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
