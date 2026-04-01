"""
tools/search.py
---------------
Stubs for the retrieval tools used by the agent pipeline.

Each function has the correct signature and return type so the orchestrator
can import and call them immediately.  Replace the mock bodies with real
implementations (Tavily, SerpAPI, ChromaDB, Pinecone, etc.) as needed.
"""

from __future__ import annotations


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


async def vector_store_search(query: str, top_k: int = 2) -> list[dict]:
    """
    Perform a semantic similarity search against the local ChromaDB collection.

    Returns a list of chunk dicts with the keys:
        - content   (str)   — raw document text
        - source    (str)   — originating document identifier
        - title     (str)   — human-readable document title
        - year      (int)   — publication / report year
        - category  (str)   — thematic category for downstream filtering
    """
    from tools.vector_store import get_retriever  # local import avoids circular deps at startup

    retriever = get_retriever(k=top_k)
    docs = await retriever.ainvoke(query)

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "title": doc.metadata.get("title", ""),
            "year": doc.metadata.get("year"),
            "category": doc.metadata.get("category", ""),
        }
        for doc in docs
    ]
