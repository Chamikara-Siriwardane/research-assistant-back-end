"""
agents/graph.py
---------------
Assembles the LangGraph StateGraph for the multi-agent research pipeline
and exposes a single module-level `compiled_graph` for use by the
orchestrator and any test fixtures.

Topology
--------

                    ┌─────────────────┐
          START ──► │   supervisor    │
                    └────────┬────────┘
                             │  _supervisor_router  (conditional)
               ┌─────────────┼──────────────┬─────────────────┐
               ▼             ▼              ▼                 ▼
          librarian        scout         analyst        synthesizer ──► END
               │             │              │
               └─────────────┴──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     critic      │
                    └────────┬────────┘
                             │  _critic_router  (conditional)
                    ┌────────┴────────┐
                    ▼                 ▼
              synthesizer        supervisor
                    │             (retry loop)
                    ▼
                   END
"""

from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    analyst_node,
    critic_node,
    librarian_node,
    scout_node,
    supervisor_node,
    synthesizer_node,
)
from agents.state import AgentState


# ---------------------------------------------------------------------------
# Conditional edge routing functions
# ---------------------------------------------------------------------------

def _supervisor_router(state: AgentState) -> str:
    """Map the Supervisor's route_command to the corresponding node name."""
    routing_map = {
        "route_to_rag": "librarian",
        "route_to_web": "scout",
        "route_to_code": "analyst",
        "route_to_synthesizer": "synthesizer",
    }
    # Fall back to supervisor if an unexpected command is returned, preventing
    # the graph from entering an undefined state.
    return routing_map.get(state["route_command"], "supervisor")


# Maximum number of retrieval-critique cycles before forcing synthesis.
_MAX_RETRIES = 3


def _critic_router(state: AgentState) -> str:
    """
    Route to the Synthesizer when context is valid or the retry cap is hit;
    loop back to the Supervisor otherwise for a different retrieval strategy.
    """
    if state["is_valid"] or state.get("retry_count", 0) >= _MAX_RETRIES:
        return "synthesizer"
    return "supervisor"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """
    Construct, wire, and compile the research pipeline StateGraph.

    Returns the compiled graph object, which can be used with:
        - compiled_graph.ainvoke(state)        — full execution, returns final state
        - compiled_graph.astream(state)        — yields state snapshots per node
        - compiled_graph.astream_events(state) — granular LangChain callback events
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────
    builder.add_node("supervisor",   supervisor_node)
    builder.add_node("librarian",    librarian_node)
    builder.add_node("scout",        scout_node)
    builder.add_node("analyst",      analyst_node)
    builder.add_node("critic",       critic_node)
    builder.add_node("synthesizer",  synthesizer_node)

    # ── Entry point ────────────────────────────────────────────────────────
    builder.add_edge(START, "supervisor")

    # ── Supervisor → specialist  (conditional) ─────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        _supervisor_router,
        {
            "librarian":   "librarian",
            "scout":       "scout",
            "analyst":     "analyst",
            "synthesizer": "synthesizer",
        },
    )

    # ── Specialists → Critic  (always) ────────────────────────────────────
    builder.add_edge("librarian", "critic")
    builder.add_edge("scout",     "critic")
    builder.add_edge("analyst",   "critic")

    # ── Critic → synthesizer or retry  (conditional) ──────────────────────
    builder.add_conditional_edges(
        "critic",
        _critic_router,
        {
            "synthesizer": "synthesizer",
            "supervisor":  "supervisor",   # retry loop
        },
    )

    # ── Synthesizer → END ─────────────────────────────────────────────────
    builder.add_edge("synthesizer", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph — import this in the orchestrator and tests
# ---------------------------------------------------------------------------

compiled_graph = build_graph()
