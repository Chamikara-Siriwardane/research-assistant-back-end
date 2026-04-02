"""
tools/search.py
---------------
Stubs for the retrieval tools used by the agent pipeline.

Each function has the correct signature and return type so the orchestrator
can import and call them immediately.  Replace the mock bodies with real
implementations (Tavily, SerpAPI, ChromaDB, Pinecone, etc.) as needed.
"""

from __future__ import annotations

import asyncio


# ---------------------------------------------------------------------------
# Web search tool
# ---------------------------------------------------------------------------


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for documents relevant to *query*.

    Returns a list of result dicts with at minimum the keys:
        - title (str)
        - url   (str)
        - snippet (str)

    TODO: integrate a real provider (Tavily, SerpAPI, Brave Search…).
    """
    await asyncio.sleep(0)

    # Mock response — remove once a real client is wired up
    return [
        {
            "title": f"Mock Web Result {i + 1} for: {query}",
            "url": f"https://example.com/result-{i + 1}",
            "snippet": f"This is a placeholder snippet for result {i + 1}.",
        }
        for i in range(max_results)
    ]


# ---------------------------------------------------------------------------
# Vector store retrieval tool
# ---------------------------------------------------------------------------


async def vector_store_search(chat_id: int, query: str, top_k: int = 2) -> list[dict]:
    """
    Perform a chat-scoped semantic similarity search against the local ChromaDB collection.

    Returns chunk dicts containing:
        - id (str)
        - content (str)
        - metadata (dict)
        - distance (float | None)
    """
    from services.vector_store import query_chat_documents

    await asyncio.sleep(0)
    return query_chat_documents(query_text=query, chat_id=chat_id, n_results=top_k)
