"""
Supervisor node implementation.
"""

import logging
import textwrap

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.nodes.common import ainvoke_with_retry, build_llm, last_human_query
from agents.state import AgentState, RouteCommand

log = logging.getLogger("agents.supervisor")

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
    has_documents: bool = state.get("has_documents", False)

    # ── Build a retry-aware context block ──────────────────────────────────
    # On the first pass retrieved_context is empty and route_command still
    # holds the seed default, so the prompt naturally omits the retry section.
    existing_context = state.get("retrieved_context", [])
    previous_route = state.get("route_command", "")

    retry_section = ""
    if existing_context:
        tried_via = _ROUTE_TO_SPECIALIST.get(previous_route, previous_route)
        snippets = "\n".join(f"- {s[:120]}" for s in existing_context[:6])
        retry_section = textwrap.dedent(f"""

            ## ⚠️ Retry Context

            A previous retrieval attempt was judged **INSUFFICIENT** by the Critic.

            **Previously tried route:** {tried_via}

            **Context retrieved so far:**
            {snippets}

            You **MUST** choose a **different** route this time, or `route_to_synthesizer`
            only if the existing context is actually good enough despite the Critic's objection.
        """).rstrip()

    # ── Conversation history summary (keep token-light) ───────────────────
    history_lines: list[str] = []
    for msg in state["messages"][:-1]:          # everything except the latest
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Truncate each turn to avoid blowing the context window
        history_lines.append(f"{role}: {msg.content[:200]}")
    history_block = (
        "\n".join(history_lines[-6:]) if history_lines else "(first message)"
    )

    doc_status = (
        "The user **HAS** uploaded documents to this chat (`has_documents=True`). "
        "Strongly prefer `route_to_rag` unless the query clearly requires live web data or code execution."
        if has_documents else
        "The user has **NOT** uploaded any documents to this chat (`has_documents=False`). "
        "Do **NOT** use `route_to_rag`."
    )

    system_prompt = SystemMessage(
        content=textwrap.dedent(f"""
            ## Role

            You are the **Supervisor** of a PhD-caliber research assistant named **Jarvis**.
            Analyse the user's latest query together with the conversation history and decide
            which specialist agent to invoke next.

            ## Available Routes

            | Route | Specialist | Use when |
            |---|---|---|
            | `route_to_rag` | **Librarian** | Query refers to the user's private uploaded documents (PDFs) in the vector store |
            | `route_to_web` | **Scout** | Query needs up-to-date facts, news, current events, or publicly available information |
            | `route_to_code` | **Analyst** | Query involves maths, statistics, data manipulation, or any task best solved with code |
            | `route_to_synthesizer` | *(skip retrieval)* | Enough context already exists in state to write a final answer |

            ## Routing Guidelines

            - Choose `route_to_rag` when the query **clearly refers to uploaded documents** or the private knowledge base.
            - Choose `route_to_web` when the query needs **real-time or public** information.
            - Choose `route_to_code` when the query involves **computation or data tasks**.
            - Choose `route_to_synthesizer` **only** when `retrieved_context` is already rich enough to fully answer the query.
            - {doc_status}
            {retry_section}

            ## Output

            Return ONLY valid JSON matching the required schema.
        """).strip()
    )

    user_prompt = HumanMessage(
        content=(
            f"Conversation history:\n{history_block}\n\n"
            f"Latest query:\n{query}"
        )
    )

    decision: RouterDecision | None = await ainvoke_with_retry(
        router_llm, [system_prompt, user_prompt]
    )

    # Guard: if the model fails to return parseable structured output, default
    # to web search (never RAG — there may be no documents).
    route = decision.route if decision is not None else "route_to_web"

    # Hard override: never route to RAG when there are no uploaded documents,
    # regardless of what the LLM decided. The prompt instructs against it but
    # the LLM can ignore instructions on retry cycles.
    if route == "route_to_rag" and not has_documents:
        route = "route_to_web"
        log.warning(
            "Supervisor overrode route_to_rag → route_to_web (has_documents=False)"
        )

    return {
        "current_agent": "supervisor",
        "route_command": route,
    }
