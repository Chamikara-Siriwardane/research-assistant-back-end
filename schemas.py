from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inbound
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
