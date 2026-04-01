"""
api/chat.py
-----------
POST /api/chat — Server-Sent Events streaming endpoint.

The route validates the incoming request, hands the query to the agent
orchestrator, and streams every yielded SSE event directly to the client.
"""

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from agents.orchestrator import run_research_pipeline
from schemas import ChatRequest, ErrorEvent

import json

router = APIRouter()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """
    Accept a research query and stream back Server-Sent Events.

    Each event is a JSON object with a *type* field:
    - ``"thought"`` — an intermediate reasoning step.
    - ``"text"``    — a single streamed answer token.
    - ``"error"``   — a fatal pipeline error.
    """
    return StreamingResponse(
        _event_stream(body.query),
        media_type="text/event-stream",
        headers={
            # Prevent proxies / nginx from buffering the stream
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _event_stream(query: str) -> AsyncGenerator[str, None]:
    """Wrap the orchestrator generator with a terminal [DONE] sentinel."""
    try:
        async for chunk in run_research_pipeline(query):
            yield chunk
    except Exception as exc:  # noqa: BLE001
        error_payload = json.dumps(ErrorEvent(content=str(exc)).model_dump())
        yield f"data: {error_payload}\n\n"
    finally:
        # Signal to the client that the stream has ended
        yield "data: [DONE]\n\n"
