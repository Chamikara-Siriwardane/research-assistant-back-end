"""
Librarian node implementation.
"""

from agents.nodes.common import last_human_query
from agents.state import AgentState
from services.vector_store import query_chat_documents


async def librarian_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks from ChromaDB scoped to the active chat."""
    import asyncio

    query = last_human_query(state)
    chat_id = state["chat_id"]

    results = await asyncio.to_thread(
        query_chat_documents, query_text=query, chat_id=chat_id, n_results=5,
    )

    docs = [
        f"[RAG Doc {i}] {r['content']}"
        for i, r in enumerate(results, start=1)
        if r.get("content")
    ]

    # If ChromaDB returned nothing, add an explicit note so downstream
    # agents know that no private documents were found.
    if not docs:
        docs = [f"[RAG] No uploaded documents found for chat {chat_id}."]

    return {
        "current_agent": "librarian",
        "retrieved_context": state["retrieved_context"] + docs,
    }
