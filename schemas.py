from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inbound — original chat pipeline
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Payload sent by the frontend for every chat turn."""

    query: str = Field(..., min_length=1, max_length=4096, description="The user's research question.")


# ---------------------------------------------------------------------------
# SSE event shapes (used for documentation / type-safety; serialised to JSON)
# ---------------------------------------------------------------------------


class ThoughtEvent(BaseModel):
    """An intermediate reasoning step surfaced to the UI."""

    type: Literal["thought"] = "thought"
    content: str


class TextEvent(BaseModel):
    """A single streamed token from the final LLM answer."""

    type: Literal["text"] = "text"
    content: str


class ErrorEvent(BaseModel):
    """Signals a fatal error inside the agent pipeline."""

    type: Literal["error"] = "error"
    content: str


# ---------------------------------------------------------------------------
# Chat Management schemas
# ---------------------------------------------------------------------------


class ChatCreate(BaseModel):
    """Payload for creating a new chat session."""

    title: str = Field(default="New Chat", max_length=255)


class ChatSummary(BaseModel):
    """Lightweight representation of a chat for list views."""

    id: int
    title: str
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    """Serialised message record."""

    id: int
    chat_id: int
    sender_type: str
    content: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    """Serialised document record."""

    id: int
    chat_id: int
    file_name: str
    s3_url: str
    status: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ChatDetail(BaseModel):
    """Full chat with associated messages and documents."""

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut] = []
    documents: list[DocumentOut] = []

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Document handling schemas
# ---------------------------------------------------------------------------


class DocumentStatusOut(BaseModel):
    """Current processing status of a document."""

    id: int
    status: str

    model_config = {"from_attributes": True}


class PresignedUrlOut(BaseModel):
    """Presigned S3 URL for downloading/previewing a document."""

    document_id: int
    url: str


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------


class MessageCreate(BaseModel):
    """Payload for sending a user message."""

    content: str = Field(..., min_length=1, max_length=4096)
