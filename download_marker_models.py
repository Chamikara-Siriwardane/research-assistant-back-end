"""
download_marker_models.py
--------------------------
Pre-downloads all marker-pdf model weights to the local cache.

Run this once:
    python download_marker_models.py

After it completes, test_rag_pipeline.py (and the app itself) will load
models instantly from cache without touching the network.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("download_marker_models")

if __name__ == "__main__":
    logger.info("Starting marker-pdf model download — this only needs to run once.")
    logger.info("Models will be cached at: C:\\Users\\USER\\AppData\\Local\\datalab\\datalab\\Cache")

    from marker.models import create_model_dict

    logger.info("Downloading / verifying all marker-pdf models...")
    create_model_dict()
    logger.info("All models downloaded and cached successfully.")
