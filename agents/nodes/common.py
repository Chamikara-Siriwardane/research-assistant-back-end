"""
Shared helpers used by individual node modules.
"""

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.state import AgentState
from core.config import settings


def build_llm(streaming: bool = False) -> ChatGoogleGenerativeAI:
    """Construct a ChatGoogleGenerativeAI instance with project-wide defaults."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        streaming=streaming,
    )


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
