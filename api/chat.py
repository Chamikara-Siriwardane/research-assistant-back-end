"""
api/chat.py
-----------
POST /api/chats/{chat_id}/messages/stream — Server-Sent Events streaming endpoint.

Execution pipeline:
  1. Save user message to SQLite.
  2. Fetch the sliding-window history (last 6 messages).
  3. Convert DB rows → LangChain message objects.
  4. Invoke the LangGraph pipeline & stream SSE events.
  5. Save the final AI response to SQLite.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agents.orchestrator import run_research_pipeline
from database import get_db
from models import Chat, Message
from schemas import ErrorEvent, StreamMessageRequest

import json

DbSession = Annotated[Session, Depends(get_db)]

router = APIRouter()

# Maximum number of past messages fetched for the sliding window
_WINDOW_SIZE = 6


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/chats/{chat_id}/messages/stream")
async def stream_message(
    chat_id: int,
    body: StreamMessageRequest,
    db: DbSession,
) -> StreamingResponse:
    """
    Accept a user message, persist it, run the research pipeline with
    sliding-window history, stream SSE events, and persist the AI reply.
    """
    # Validate that the chat exists
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    # ── Step 1: Save user message ─────────────────────────────────────────
    user_msg = Message(
        chat_id=chat_id,
        sender_type="user",
        content=body.content,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(user_msg)
    db.commit()

    # ── Step 2: Fetch sliding-window history ──────────────────────────────
    recent_rows = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.timestamp.desc())
        .limit(_WINDOW_SIZE)
        .all()
    )
    # Reverse so the list is chronological (oldest → newest)
    recent_rows.reverse()

    # ── Step 3: Format for LangChain ─────────────────────────────────────
    from langchain_core.messages import AIMessage, HumanMessage

    history: list[HumanMessage | AIMessage] = []
    for row in recent_rows:
        if row.sender_type == "user":
            history.append(HumanMessage(content=row.content))
        elif row.sender_type == "jarvis":
            history.append(AIMessage(content=row.content))

    return StreamingResponse(
        _event_stream(history, chat_id, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _event_stream(
    history: list,
    chat_id: int,
    db: Session,
) -> AsyncGenerator[str, None]:
    """
    Run the orchestrator, yield SSE chunks, and persist the final AI answer.
    """
    final_text_parts: list[str] = []

    try:
        async for chunk in run_research_pipeline(history, chat_id):
            yield chunk

            # Collect streamed text tokens for the final AI message
            try:
                line = chunk.strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    payload = json.loads(line[len("data: "):])
                    if payload.get("type") == "text":
                        final_text_parts.append(payload["content"])
            except (json.JSONDecodeError, KeyError):
                pass

    except Exception as exc:  # noqa: BLE001
        error_payload = json.dumps(ErrorEvent(content=str(exc)).model_dump())
        yield f"data: {error_payload}\n\n"
    finally:
        # ── Step 5: Save AI response ─────────────────────────────────────
        final_text = "".join(final_text_parts)
        if final_text:
            ai_msg = Message(
                chat_id=chat_id,
                sender_type="jarvis",
                content=final_text,
                timestamp=datetime.now(timezone.utc),
            )
            db.add(ai_msg)
            db.commit()

        yield "data: [DONE]\n\n"
