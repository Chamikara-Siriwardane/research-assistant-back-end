"""
Supervisor node implementation.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.nodes.common import build_llm, last_human_query
from agents.state import AgentState, RouteCommand


class RouterDecision(BaseModel):
    """Output schema for the Supervisor's routing decision."""

    route: RouteCommand = Field(
        description=(
            "Which specialist agent to invoke next. Choose 'route_to_synthesizer' "
            "only when the retrieved_context is already sufficient."
        )
    )
    reasoning: str = Field(description="One-sentence justification for the routing choice.")


# Maps route_command values to the specialist that ran
_ROUTE_TO_SPECIALIST: dict[str, str] = {
    "route_to_rag": "librarian (private document store / RAG)",
    "route_to_web": "scout (web search)",
    "route_to_code": "analyst (code execution)",
}


async def supervisor_node(state: AgentState) -> dict:
    """Analyze the latest query and route to the next specialist node."""
    llm = build_llm()
    router_llm = llm.with_structured_output(RouterDecision)

    query = last_human_query(state)

    # ── Build a retry-aware context block ──────────────────────────────────
    # On the first pass retrieved_context is empty and route_command still
    # holds the seed default, so the prompt naturally omits the retry section.
    existing_context = state.get("retrieved_context", [])
    previous_route = state.get("route_command", "")

    retry_section = ""
    if existing_context:
        tried_via = _ROUTE_TO_SPECIALIST.get(previous_route, previous_route)
        snippets = "\n".join(f"  • {s[:120]}" for s in existing_context[:6])
        retry_section = (
            "\n\n⚠️  RETRY CONTEXT — a previous retrieval attempt was judged "
            "INSUFFICIENT by the Critic.\n"
            f"Previously tried route: {tried_via}\n"
            f"Context retrieved so far:\n{snippets}\n\n"
            "You MUST choose a DIFFERENT route this time, or route_to_synthesizer "
            "only if the existing context is actually good enough despite the "
            "Critic's objection."
        )

    # ── Conversation history summary (keep token-light) ───────────────────
    history_lines: list[str] = []
    for msg in state["messages"][:-1]:          # everything except the latest
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Truncate each turn to avoid blowing the context window
        history_lines.append(f"{role}: {msg.content[:200]}")
    history_block = (
        "\n".join(history_lines[-6:]) if history_lines else "(first message)"
    )

    system_prompt = SystemMessage(
        content=(
            "You are the supervisor of a PhD-caliber research assistant named Jarvis. "
            "Your job is to analyse the user's latest query together with the "
            "conversation history and decide which specialist agent to invoke next.\n\n"
            "Available routes:\n"
            "  route_to_rag  → Librarian – searches the user's private uploaded "
            "documents (PDFs) stored in a vector database.\n"
            "  route_to_web  → Scout – performs live web searches for recent, "
            "real-time, or news-based information.\n"
            "  route_to_code → Analyst – executes code for data analysis, "
            "calculations, or programmatic tasks.\n"
            "  route_to_synthesizer → skip retrieval; enough context already "
            "exists in state to write a final answer.\n\n"
            "Guidelines:\n"
            "• Choose route_to_rag when the query clearly refers to the user's "
            "uploaded documents or private knowledge base.\n"
            "• Choose route_to_web when the query needs up-to-date facts, news, "
            "current events, or publicly available information.\n"
            "• Choose route_to_code when the query involves maths, statistics, "
            "data manipulation, or any task best solved with code.\n"
            "• Choose route_to_synthesizer ONLY when the retrieved_context is "
            "already rich enough to fully answer the query.\n"
            "• When in doubt between RAG and Web, prefer RAG if the user has "
            "uploaded relevant documents for this chat.\n"
            f"{retry_section}\n\n"
            "Return ONLY valid JSON matching the required schema."
        )
    )

    user_prompt = HumanMessage(
        content=(
            f"Conversation history:\n{history_block}\n\n"
            f"Latest query:\n{query}"
        )
    )

    decision: RouterDecision = await router_llm.ainvoke(
        [system_prompt, user_prompt]
    )

    return {
        "current_agent": "supervisor",
        "route_command": decision.route,
    }
