"""
agents/orchestrator.py
----------------------
Public interface between the FastAPI streaming route and the LangGraph graph.

`run_research_pipeline` is an async generator that:
  1. Feeds the user query into the compiled graph via `astream_events`.
  2. Translates granular LangGraph / LangChain callback events into the
     SSE-formatted strings that `api/chat.py` forwards to the browser.

Event mapping
-------------
  on_chain_start  (agent nodes)      → ThoughtEvent  ("Agent X is working…")
  on_chat_model_stream (synthesizer) → TextEvent     (streamed answer token)
  exceptions                         → ErrorEvent
  graph end                          → [DONE] sentinel  (emitted by api/chat.py)
"""

import json
from collections.abc import AsyncGenerator

from langchain_core.messages import BaseMessage

from agents.graph import compiled_graph
from agents.state import AgentState
from schemas import ErrorEvent, TextEvent, ThoughtEvent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Node names that should produce a visible "thought" event in the UI
_AGENT_NODES: frozenset[str] = frozenset(
    {"supervisor", "librarian", "scout", "analyst", "critic", "synthesizer"}
)

# Human-readable status label for each node
_NODE_LABELS: dict[str, str] = {
    "supervisor":   "Supervisor is routing the query…",
    "librarian":    "Librarian is searching the document store…",
    "scout":        "Scout is browsing the web…",
    "analyst":      "Analyst is running the code…",
    "critic":       "Critic is evaluating the retrieved context…",
    "synthesizer":  "Synthesizer is writing the final answer…",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sse(event_model) -> str:
    """Serialise a Pydantic event model into an SSE data line."""
    return f"data: {json.dumps(event_model.model_dump())}\n\n"


# ---------------------------------------------------------------------------
# Public async generator — consumed by api/chat.py
# ---------------------------------------------------------------------------

async def run_research_pipeline(
    messages: list[BaseMessage],
    chat_id: int,
    has_documents: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Drive the LangGraph pipeline and yield SSE strings for every meaningful
    event.

    Parameters
    ----------
    messages : list[BaseMessage]
        Sliding-window conversation history (already formatted as LangChain
        message objects by the endpoint).
    chat_id : int
        Active chat session — forwarded into the graph state so the
        Librarian can scope its ChromaDB queries.
    has_documents : bool
        Pre-computed (and cached) flag indicating whether the chat has at
        least one ready document.  Passed straight into AgentState so the
        Supervisor can make a data-driven routing decision without touching
        the DB itself.

    Yields
    ------
    SSE-formatted strings of the form:
        data: <json_payload>\\n\\n
    """
    initial_state: AgentState = {
        "messages":          messages,
        "chat_id":           chat_id,
        "current_agent":     "",
        "retrieved_context": [],
        "retrieved_pages":   [],
        "is_valid":          False,
        "retry_count":       0,
        "route_command":     "route_to_rag",  # overwritten immediately by supervisor
        "has_documents":     has_documents,
    }

    try:
        async for event in compiled_graph.astream_events(initial_state, version="v2"):
            event_type: str = event["event"]
            # `langgraph_node` in metadata tells us which node fired this event
            node_name: str = event.get("metadata", {}).get("langgraph_node", "")

            # ── Node start → surface a thought event ──────────────────────
            # Check event["name"] == node_name so sub-runnables within the
            # node (LLM chain, structured-output wrapper, etc.) don't each
            # fire a duplicate thought — only the top-level node entry does.
            if (
                event_type == "on_chain_start"
                and node_name in _AGENT_NODES
                and event.get("name") == node_name
            ):
                label = _NODE_LABELS.get(node_name, f"{node_name.title()} is working…")
                yield _sse(ThoughtEvent(content=label))

            # ── Streamed token from Synthesizer → text event ──────────────
            # We intentionally filter to 'synthesizer' only so that the
            # Supervisor and Critic's internal LLM calls stay invisible to
            # the end user (they already produced ThoughtEvents above).
            elif (
                event_type == "on_chat_model_stream"
                and node_name == "synthesizer"
            ):
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    yield _sse(TextEvent(content=chunk.content))

    except Exception as exc:  # noqa: BLE001
        yield _sse(ErrorEvent(content=f"Pipeline error: {exc}"))
