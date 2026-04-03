"""
Shared helpers used by individual node modules.
"""

import asyncio
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.state import AgentState
from core.config import settings

# Maximum LLM call attempts before giving up on a single node invocation.
_MAX_ATTEMPTS = 4


def build_llm(streaming: bool = False) -> ChatGoogleGenerativeAI:
    """Construct a ChatGoogleGenerativeAI instance with project-wide defaults."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        streaming=streaming,
    )


def _parse_retry_delay(exc: Exception) -> float:
    """
    Extract the suggested retry delay (seconds) from a Gemini 429 error.
    Falls back to 65 seconds if the field cannot be parsed.
    """
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(exc))
    return float(match.group(1)) + 5 if match else 65.0


def _is_rate_limit(exc: Exception) -> bool:
    """Return True for 429 / quota-exceeded errors from the Gemini API."""
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "resource_exhausted" in msg


async def ainvoke_with_retry(chain: Any, messages: list) -> Any:
    """
    Call chain.ainvoke(messages) and retry on Gemini rate-limit errors.

    Waits exactly as long as the API specifies (+ 5 s buffer) before each
    retry so we never spin faster than the quota window allows.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await chain.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit(exc) and attempt < _MAX_ATTEMPTS:
                delay = _parse_retry_delay(exc)
                await asyncio.sleep(delay)
                last_exc = exc
            else:
                raise
    raise last_exc  # type: ignore[misc]


async def astream_with_retry(llm: Any, messages: list):
    """
    Async generator that streams tokens from llm.astream(messages) and
    transparently retries the entire stream on rate-limit errors.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            async for chunk in llm.astream(messages):
                yield chunk
            return
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit(exc) and attempt < _MAX_ATTEMPTS:
                delay = _parse_retry_delay(exc)
                await asyncio.sleep(delay)
                last_exc = exc
            else:
                raise
    raise last_exc  # type: ignore[misc]


def last_human_query(state: AgentState) -> str:
    """Return the latest human message from state or an empty string."""
    return next(
        (
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        "",
    )
