"""
main.py
-------
Application entry point.

Run locally with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.chats import router as chats_router
from api.documents import router as documents_router
from api.messages import router as messages_router
from core.config import settings
from database import Base, engine

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
