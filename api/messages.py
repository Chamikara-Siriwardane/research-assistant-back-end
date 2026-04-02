"""
api/messages.py
---------------
Core AI engine endpoint — SSE streaming for agent thoughts and answer tokens.

Router prefix: /api/messages
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from models import Chat, Message
from schemas import MessageCreate

router = APIRouter(prefix="/messages", tags=["messages"])


async def _mock_agent_stream(user_content: str) -> AsyncGenerator[str, None]:
    """
    Yield a mock stream of JSON SSE events simulating the agent-thoughts
    pipeline.  Replace with real LangGraph orchestration later.

    Stream sequence:
      1. Routing Agent thought
      2. RAG Agent thought
      3. Token-by-token answer
    """
    # Step 1 — Routing agent thought
    yield f"data: {json.dumps({'type': 'thought', 'content': '[Routing Agent] Analyzing request...'})}\n\n"
    await asyncio.sleep(1)

    # Step 2 — RAG agent thought
    yield f"data: {json.dumps({'type': 'thought', 'content': '[RAG Agent] Searching vector database...'})}\n\n"
    await asyncio.sleep(1)

    # Step 3 — Streamed answer tokens
    tokens = ["Based ", "on the ", "document..."]
    for token in tokens:
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # Signal end of stream
    yield "data: [DONE]\n\n"


@router.post("/chats/{chat_id}/messages/stream")
async def stream_message(
    chat_id: int,
    body: MessageCreate,
    db: Session = Depends(get_db),
):
    """
    Accept a user message, persist it, and stream back SSE events
    containing agent thoughts and answer tokens.
    """
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Persist the user message
    user_message = Message(
        chat_id=chat_id,
        sender_type="user",
        content=body.content,
    )
    db.add(user_message)

    # Update chat title from first message if still default
    if chat.title == "New Chat":
        chat.title = body.content[:50]

    chat.updated_at = datetime.now(timezone.utc)
    db.commit()

    return StreamingResponse(
        _mock_agent_stream(body.content),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
