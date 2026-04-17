"""
Critic node implementation.
"""

import textwrap

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.nodes.common import ainvoke_with_retry, build_llm, last_human_query
from agents.state import AgentState


class CriticDecision(BaseModel):
    """Output schema for the Critic's validation verdict."""

    is_valid: bool = Field(
        description=(
            "True if the retrieved context is relevant and sufficient to answer "
            "the query in full. False if it is vague, off-topic, or incomplete."
        )
    )
    reasoning: str = Field(description="Brief explanation of the verdict.")


async def critic_node(state: AgentState) -> dict:
    """Validate whether retrieved context is sufficient for final synthesis."""
    llm = build_llm()
    critic_llm = llm.with_structured_output(CriticDecision)

    query = last_human_query(state)
    context_block = "\n".join(state["retrieved_context"]) or "No context retrieved yet."

    # ── Conversation history for multi-turn awareness ─────────────────────
    history_lines: list[str] = []
    for msg in state["messages"][:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content[:200]}")
    history_block = (
        "\n".join(history_lines[-6:]) if history_lines else "(first message)"
    )

    system_prompt = SystemMessage(
        content=textwrap.dedent("""
            ## Role

            You are the **Critic** — a rigorous quality-control judge for **Jarvis**, a PhD-caliber
            research assistant. Your job is to decide whether the retrieved context is sufficient
            and directly relevant to answer the user's latest query, keeping the ongoing conversation
            in mind.

            ## Mark VALID (`is_valid=true`) when

            - The context contains concrete, on-topic information that can answer the query.
            - Even partial coverage is acceptable if the **key points** are addressed.
            - The context contains a `[LIBRARIAN SUMMARY]` line confirming that PDF pages were
              fetched — the Synthesizer receives those pages as full multimodal PDF input, so the
              short text snippets here are only a preview. **Always mark VALID in this case.**

            ## Mark INVALID (`is_valid=false`) when

            - The context is empty, completely off-topic, or only contains boilerplate/placeholder text.
            - Critical aspects of the query are entirely unaddressed **and** no `[LIBRARIAN SUMMARY]`
              line is present.

            ## Important

            Do **not** be overly strict. A good-enough answer is better than an infinite retry loop.
            If the context covers the core of the query, mark it **VALID** and let the Synthesizer
            refine the answer.

            ## Output

            Return ONLY valid JSON matching the required schema.
        """).strip()
    )
    user_prompt = HumanMessage(
        content=(
            f"Conversation history:\n{history_block}\n\n"
            f"Latest user query:\n{query}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            "Is this context sufficient and relevant to answer the query?"
        )
    )

    decision: CriticDecision | None = await ainvoke_with_retry(
        critic_llm, [system_prompt, user_prompt]
    )

    # Guard: if the model fails to return parseable structured output, treat
    # context as valid so the graph proceeds to the Synthesizer rather than
    # crashing or looping.
    is_valid = decision.is_valid if decision is not None else True

    return {
        "current_agent": "critic",
        "is_valid": is_valid,
        "retry_count": state.get("retry_count", 0) + 1,
    }
