"""
api/chats.py
------------
Chat Management endpoints — CRUD operations for chat sessions.

Router prefix: /api/chats
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from database import get_db
from models import Chat
from schemas import ChatCreateOut, ChatDetail, ChatSummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("/", response_model=list[ChatSummary])
def list_chats(request: Request, db: Annotated[Session, Depends(get_db)]):
    """Return a lightweight list of all chats ordered by newest first."""
    logger.info(
        "[GET /api/chats/] Request received from client=%s",
        request.client.host if request.client else "unknown",
    )
    chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
    logger.info("[GET /api/chats/] Returning %d chats", len(chats))
    if chats:
        logger.debug(
            "[GET /api/chats/] Top chat IDs (max 5): %s",
            [chat.id for chat in chats[:5]],
        )
    return chats


@router.post("/", response_model=ChatCreateOut, status_code=201)
def create_chat(request: Request, db: Annotated[Session, Depends(get_db)]):
    """Create a new empty chat session and return the chat_id."""
    logger.info(
        "[POST /api/chats/] Create chat request received from client=%s",
        request.client.host if request.client else "unknown",
    )
    chat = Chat()
    db.add(chat)
    db.commit()
    db.refresh(chat)
    logger.info(
        "[POST /api/chats/] Chat created chat_id=%s title=%s",
        chat.id,
        chat.title,
    )
    return ChatCreateOut(chat_id=chat.id, title=chat.title, created_at=chat.created_at)


@router.get("/{chat_id}", response_model=ChatDetail, responses={404: {"description": "Chat not found."}})
def get_chat(chat_id: int, db: Annotated[Session, Depends(get_db)]):
    """Return chat details including all messages and documents."""
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat
