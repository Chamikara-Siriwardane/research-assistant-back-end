"""
agents/state.py
---------------
Defines the shared state schema that flows through every node in the
LangGraph research pipeline.

Each node receives the *full* state and returns a **partial** dict
containing only the keys it modifies.  LangGraph merges that partial
update back into the canonical state before passing it to the next node.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Route command literal — emitted by the Supervisor node
# ---------------------------------------------------------------------------

RouteCommand = Literal[
    "route_to_rag",          # Query needs a private document store
    "route_to_web",          # Query needs recent / real-time web data
    "route_to_code",         # Query needs data analysis or code execution
    "route_to_synthesizer",  # Context is already sufficient; skip retrieval
]

# ---------------------------------------------------------------------------
# Retrieved PDF page — emitted by the Librarian node
# ---------------------------------------------------------------------------

class RetrievedPage(TypedDict):
    """A single PDF page fetched from S3 after a ChromaDB vector match."""
    document_id: int    # FK to the Document row in SQLite
    page_number: int    # 1-based page index within the source PDF
    page_bytes: bytes   # Raw single-page PDF byte stream

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """The single source of truth passed between every LangGraph node."""

    # Full conversation history.
    # The `add_messages` reducer *appends* new messages rather than
    # overwriting the list, so parallel branches don't clobber each other.
    messages: Annotated[list[BaseMessage], add_messages]

    # The active chat session — used by the Librarian to scope ChromaDB
    # queries and by the endpoint to persist messages.
    chat_id: int

    # Name of the node that is *currently* acting.
    # Surfaced in the UI as a "thought" event.
    current_agent: str

    # Accumulated snippets / tool outputs from specialist agents.
    # Each agent node appends its results; the Synthesizer consumes all of them.
    retrieved_context: list[str]

    # Raw PDF page bytes fetched from S3 by the Librarian node.
    # Each entry corresponds to one ChromaDB vector match and carries the
    # actual page content that the Synthesizer passes to the multimodal LLM.
    retrieved_pages: list[RetrievedPage]

    # Set by the Critic node.
    # True  → route to Synthesizer.
    # False → loop back to Supervisor for a different retrieval strategy.
    is_valid: bool

    # Incremented by the Critic after each retrieval cycle.
    # The graph forces a route to Synthesizer once this hits MAX_RETRIES,
    # preventing infinite loops when the Critic never marks context valid.
    retry_count: int

    # Set by the Supervisor node.
    # Tells the conditional edge which specialist to activate next.
    route_command: RouteCommand

    # Injected by the endpoint before the graph starts.
    # True when the chat has at least one document with status='ready'.
    # Used by the Supervisor to make a data-driven RAG routing decision
    # instead of relying purely on LLM inference.
    has_documents: bool
