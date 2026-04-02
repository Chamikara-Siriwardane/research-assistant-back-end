"""
api/messages.py
---------------
Core AI engine endpoint — SSE streaming for chat responses.

Router prefix: /api/messages
"""

import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session

from core.config import settings
from database import get_db
from models import Chat, Message
from schemas import ErrorEvent, MessageCreate, TextEvent, ThoughtEvent
from services.vector_store import query_chat_documents

router = APIRouter(prefix="/messages", tags=["messages"])


def _sse(event_model) -> str:
    """Serialise a Pydantic event model into an SSE data line."""
    return f"data: {json.dumps(event_model.model_dump())}\n\n"


async def _chat_stream(chat_id: int, user_content: str, db: Session) -> AsyncGenerator[str, None]:
    """
    Retrieve chat-scoped context from Chroma, then stream a Gemini response.
    """
    full_response = ""

    try:
        yield _sse(ThoughtEvent(content="Retrieving context from the active chat..."))
        matches = query_chat_documents(query_text=user_content, chat_id=chat_id, n_results=5)

        if matches:
            yield _sse(ThoughtEvent(content=f"Retrieved {len(matches)} relevant chunk(s) from this chat."))
            context_block = "\n\n".join(
                f"[Chunk {index}] {match['content']}\nMetadata: {json.dumps(match['metadata'])}"
                for index, match in enumerate(matches, start=1)
            )
        else:
            yield _sse(ThoughtEvent(content="No chat-scoped document chunks were found."))
            context_block = "No document context available for this chat."

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.3,
            streaming=True,
        )

        system_prompt = SystemMessage(
            content=(
                "You are Jarvis, a careful research assistant. Use the retrieved "
                "chat-scoped document context to answer the user's question. "
                "If the context is insufficient, say so plainly."
            )
        )
        user_prompt = HumanMessage(
            content=(
                f"User question:\n{user_content}\n\n"
                f"Retrieved context:\n{context_block}"
            )
        )

        async for chunk in llm.astream([system_prompt, user_prompt]):
            chunk_content = chunk.content
            if isinstance(chunk_content, list):
                chunk_content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in chunk_content
                )
            if chunk_content:
                text = str(chunk_content)
                full_response += text
                yield _sse(TextEvent(content=text))

        assistant_message = Message(
            chat_id=chat_id,
            sender_type="jarvis",
            content=full_response,
        )
        db.add(assistant_message)
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if chat:
            chat.updated_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        yield _sse(ErrorEvent(content=f"Pipeline error: {exc}"))

    finally:
        yield "data: [DONE]\n\n"


@router.post(
    "/chats/{chat_id}/messages/stream",
    responses={404: {"description": "Chat not found."}},
)
async def stream_message(
    chat_id: int,
    body: MessageCreate,
    db: Annotated[Session, Depends(get_db)],
):
    """
    Accept a user message, persist it, and stream back SSE events
    containing retrieval thoughts and answer tokens.
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
        _chat_stream(chat_id, body.content, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
