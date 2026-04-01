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
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """The single source of truth passed between every LangGraph node."""

    # Full conversation history.
    # The `add_messages` reducer *appends* new messages rather than
    # overwriting the list, so parallel branches don't clobber each other.
    messages: Annotated[list[BaseMessage], add_messages]

    # Name of the node that is *currently* acting.
    # Surfaced in the UI as a "thought" event.
    current_agent: str

    # Accumulated snippets / tool outputs from specialist agents.
    # Each agent node appends its results; the Synthesizer consumes all of them.
    retrieved_context: list[str]

    # Set by the Critic node.
    # True  → route to Synthesizer.
    # False → loop back to Supervisor for a different retrieval strategy.
    is_valid: bool

    # Set by the Supervisor node.
    # Tells the conditional edge which specialist to activate next.
    route_command: RouteCommand
