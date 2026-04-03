"""
main.py
-------
Application entry point.

Run locally with:
    uvicorn main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.chats import router as chats_router
from api.documents import router as documents_router
from api.messages import router as messages_router
from core.config import settings
from database import Base, engine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_LEVEL = getattr(logging, settings.log_level.upper(), logging.INFO)


def _ensure_root_stream_handler() -> None:
    """Attach a console handler for app logs when uvicorn does not configure root."""
    root_logger = logging.getLogger()
    has_stream_handler = any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers)

    if has_stream_handler:
        return

    handler = logging.StreamHandler()
    handler.setLevel(_LOG_LEVEL)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root_logger.addHandler(handler)


def _configure_app_loggers() -> None:
    """Ensure project loggers emit at the configured level under uvicorn."""
    _ensure_root_stream_handler()
    logging.getLogger().setLevel(_LOG_LEVEL)
    logging.getLogger("uvicorn.error").setLevel(_LOG_LEVEL)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    # Explicitly enable common project namespaces.
    for logger_name in ["main", "api", "services", "agents", "tools", "core", "database"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(_LOG_LEVEL)
        logger.propagate = True


_configure_app_loggers()
logging.getLogger(__name__).info("Logging configured with level=%s", settings.log_level.upper())

# ---------------------------------------------------------------------------
# Create database tables on startup
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    debug=settings.debug,
)


@app.on_event("startup")
async def startup_logging_probe() -> None:
    """Re-apply logger levels after uvicorn startup and emit a probe message."""
    _configure_app_loggers()
    logging.getLogger("api.chats").info("Application logger probe: api.chats logger is active")

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(chat_router, prefix="/api")
app.include_router(chats_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(messages_router, prefix="/api")

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "version": settings.app_version}
