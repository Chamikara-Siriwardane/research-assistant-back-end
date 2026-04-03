"""
Librarian node implementation.
"""

import asyncio

from agents.nodes.common import last_human_query
from agents.state import AgentState


async def librarian_node(state: AgentState) -> dict:
    """Simulate local RAG retrieval and append context snippets."""
    await asyncio.sleep(0.5)
    query = last_human_query(state)
    mock_docs = [
        f"[RAG Doc 1] Relevant passage about '{query}' retrieved from the internal knowledge base.",
        "[RAG Doc 2] Supporting evidence from a stored research paper (mock).",
    ]

    return {
        "current_agent": "librarian",
        "retrieved_context": state["retrieved_context"] + mock_docs,
    }
