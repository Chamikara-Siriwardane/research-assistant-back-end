"""
test_chat.py
------------
Standalone script that replicates the POST /api/chats/{chat_id}/messages/stream
endpoint logic without FastAPI.  Run it directly:

    python test_chat.py

It will:
  1. Create a chat (or reuse an existing one).
  2. Prompt you for messages in a loop.
  3. For each message, execute the exact same 5-step pipeline the endpoint uses:
       Step 1 — save user message to SQLite
       Step 2 — fetch sliding-window history (last 6 messages)
       Step 3 — convert DB rows → LangChain messages
       Step 4 — invoke the LangGraph pipeline & print streamed events
       Step 5 — save the final AI response to SQLite
"""

import asyncio
import json
import sys
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from agents.orchestrator import run_research_pipeline
from database import Base, SessionLocal, engine
from models import Chat, Message
from schemas import ErrorEvent

# Sliding-window size — same as the endpoint
_WINDOW_SIZE = 6


# ---------------------------------------------------------------------------
# Core pipeline — mirrors api/chat.py exactly
# ---------------------------------------------------------------------------

async def run_chat_turn(chat_id: int, user_content: str) -> str:
    """
    Execute one chat turn end-to-end, identical to the streaming endpoint.

    Returns the final assembled AI response text.
    """
    db = SessionLocal()
    try:
        # ── Validate chat exists ──────────────────────────────────────────
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if chat is None:
            print(f"[ERROR] Chat {chat_id} not found.")
            sys.exit(1)

        # ── Step 1: Save user message ─────────────────────────────────────
        user_msg = Message(
            chat_id=chat_id,
            sender_type="user",
            content=user_content,
            timestamp=datetime.now(timezone.utc),
        )
        db.add(user_msg)
        db.commit()
        print(f"\n[DB] Saved user message (id={user_msg.id})")

        # ── Step 2: Fetch sliding-window history ──────────────────────────
        recent_rows = (
            db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.timestamp.desc())
            .limit(_WINDOW_SIZE)
            .all()
        )
        recent_rows.reverse()
        print(f"[DB] Fetched {len(recent_rows)} messages for sliding window")

        # ── Step 3: Format for LangChain ──────────────────────────────────
        history: list[HumanMessage | AIMessage] = []
        for row in recent_rows:
            if row.sender_type == "user":
                history.append(HumanMessage(content=row.content))
            elif row.sender_type == "jarvis":
                history.append(AIMessage(content=row.content))

        # ── Step 4: Invoke graph & stream ─────────────────────────────────
        final_text_parts: list[str] = []

        print("\n--- SSE Stream Start ---")
        try:
            async for chunk in run_research_pipeline(history, chat_id):
                # Print every SSE line as-is (just like the client would see)
                line = chunk.strip()
                if not line:
                    continue

                if line == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[len("data: "):])
                        event_type = payload.get("type", "")

                        if event_type == "thought":
                            print(f"  [THOUGHT] {payload['content']}")
                        elif event_type == "text":
                            print(payload["content"], end="", flush=True)
                            final_text_parts.append(payload["content"])
                        elif event_type == "error":
                            print(f"\n  [ERROR] {payload['content']}")
                    except (json.JSONDecodeError, KeyError):
                        pass

        except Exception as exc:
            error_payload = json.dumps(ErrorEvent(content=str(exc)).model_dump())
            print(f"\n  [PIPELINE ERROR] {exc}")
            final_text_parts.clear()

        print("\n--- SSE Stream End ---")

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
            print(f"[DB] Saved AI response (id={ai_msg.id}, length={len(final_text)})")

        return final_text

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def get_or_create_chat() -> int:
    """Prompt the user to pick an existing chat or create a new one."""
    db = SessionLocal()
    try:
        chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()

        if chats:
            print("\nExisting chats:")
            for c in chats:
                print(f"  [{c.id}] {c.title}  (updated {c.updated_at})")

        choice = input("\nEnter a chat ID to reuse, or press Enter to create a new one: ").strip()

        if choice:
            chat_id = int(choice)
            if not db.query(Chat).filter(Chat.id == chat_id).first():
                print(f"Chat {chat_id} not found — creating a new one.")
            else:
                return chat_id

        new_chat = Chat(title="Test Chat", created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc))
        db.add(new_chat)
        db.commit()
        print(f"Created new chat (id={new_chat.id})")
        return new_chat.id
    finally:
        db.close()


async def main() -> None:
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    print("=" * 60)
    print("  Jarvis Research Assistant — Chat Test Script")
    print("  (replicates POST /api/chats/{id}/messages/stream)")
    print("=" * 60)

    chat_id = get_or_create_chat()
    print(f"\nUsing chat_id={chat_id}.  Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        await run_chat_turn(chat_id, user_input)
        print()


if __name__ == "__main__":
    asyncio.run(main())
