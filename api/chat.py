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

import json
import logging
import time
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

log = logging.getLogger("api.chat")

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
    log.info("Incoming stream request | chat_id=%d | content_length=%d", chat_id, len(body.content))

    # Validate that the chat exists
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if chat is None:
        log.warning("Chat not found | chat_id=%d", chat_id)
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
    log.info("Step 1 | Saved user message | chat_id=%d | message_id=%d", chat_id, user_msg.id)

    # ── Step 2: Fetch sliding-window history ──────────────────────────────
    recent_rows = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.timestamp.desc())
        .limit(_WINDOW_SIZE)
        .all()
    )
    recent_rows.reverse()
    log.info("Step 2 | Loaded sliding window | chat_id=%d | messages_fetched=%d", chat_id, len(recent_rows))

    # ── Step 3: Format for LangChain ─────────────────────────────────────
    from langchain_core.messages import AIMessage, HumanMessage

    history: list[HumanMessage | AIMessage] = []
    for row in recent_rows:
        if row.sender_type == "user":
            history.append(HumanMessage(content=row.content))
        elif row.sender_type == "jarvis":
            history.append(AIMessage(content=row.content))

    log.info(
        "Step 3 | Formatted history | chat_id=%d | human=%d | ai=%d",
        chat_id,
        sum(1 for m in history if isinstance(m, HumanMessage)),
        sum(1 for m in history if isinstance(m, AIMessage)),
    )

    log.info("Step 4 | Starting LangGraph pipeline | chat_id=%d", chat_id)
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
    token_count = 0
    thought_count = 0
    start_time = time.monotonic()

    try:
        async for chunk in run_research_pipeline(history, chat_id):
            yield chunk

            # Collect streamed text tokens and track event counts for logging
            try:
                line = chunk.strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    payload = json.loads(line[len("data: "):])
                    event_type = payload.get("type")
                    if event_type == "text":
                        final_text_parts.append(payload["content"])
                        token_count += 1
                    elif event_type == "thought":
                        thought_count += 1
                        log.debug("SSE thought | chat_id=%d | %s", chat_id, payload.get("content", ""))
                    elif event_type == "error":
                        log.error("SSE pipeline error | chat_id=%d | %s", chat_id, payload.get("content", ""))
            except (json.JSONDecodeError, KeyError):
                pass

    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled exception in event stream | chat_id=%d", chat_id)
        error_payload = json.dumps(ErrorEvent(content=str(exc)).model_dump())
        yield f"data: {error_payload}\n\n"
    finally:
        elapsed = time.monotonic() - start_time

        # ── Step 5: Save AI response ──────────────────────────────────────
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
            log.info(
                "Step 5 | Saved AI response | chat_id=%d | message_id=%d | chars=%d",
                chat_id, ai_msg.id, len(final_text),
            )
        else:
            log.warning("Step 5 | No AI response to save | chat_id=%d", chat_id)

        log.info(
            "Stream complete | chat_id=%d | thoughts=%d | tokens=%d | elapsed=%.2fs",
            chat_id, thought_count, token_count, elapsed,
        )

        yield "data: [DONE]\n\n"

