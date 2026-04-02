"""
api/chats.py
------------
Chat Management endpoints — CRUD operations for chat sessions.

Router prefix: /api/chats
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Chat
from schemas import ChatCreate, ChatDetail, ChatSummary

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("/", response_model=list[ChatSummary])
def list_chats(db: Session = Depends(get_db)):
    """Return a lightweight list of all chats ordered by newest first."""
    chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
    return chats


@router.post("/", response_model=ChatSummary, status_code=201)
def create_chat(body: ChatCreate = ChatCreate(), db: Session = Depends(get_db)):
    """Create a new empty chat session and return the chat_id."""
    chat = Chat(title=body.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("/{chat_id}", response_model=ChatDetail)
def get_chat(chat_id: int, db: Session = Depends(get_db)):
    """Return chat details including all messages and documents."""
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat
