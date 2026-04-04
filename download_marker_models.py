"""
download_marker_models.py
--------------------------
OBSOLETE — this script is no longer used.

The ingestion pipeline has been migrated from marker-pdf (text extraction +
Markdown chunking) to native multimodal embedding. PDFs are now sliced into
individual pages with pypdf and each page is embedded directly via the Gemini
multimodal embedding API. No local model weights are required.
"""

    logger.info("All models downloaded and cached successfully.")
