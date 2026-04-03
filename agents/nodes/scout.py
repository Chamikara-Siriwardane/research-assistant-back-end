"""
Scout node implementation.
"""

import asyncio

from agents.nodes.common import last_human_query
from agents.state import AgentState


async def scout_node(state: AgentState) -> dict:
    """Simulate web retrieval and append context snippets."""
    await asyncio.sleep(0.8)
    query = last_human_query(state)
    mock_results = [
        f"[Web Result 1] Breaking research on '{query}' - arxiv.org/abs/mock-2026.",
        "[Web Result 2] Recent survey article covering key findings (mock).",
        "[Web Result 3] Official documentation snippet (mock).",
    ]

    return {
        "current_agent": "scout",
        "retrieved_context": state["retrieved_context"] + mock_results,
    }
