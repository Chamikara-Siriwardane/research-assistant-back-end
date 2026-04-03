"""
Critic node implementation.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.nodes.common import build_llm, last_human_query
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
        content=(
            "You are a rigorous quality-control critic for a research assistant "
            "named Jarvis. Your job is to decide whether the retrieved context is "
            "sufficient and directly relevant to answer the user's latest query, "
            "keeping the ongoing conversation in mind.\n\n"
            "Mark VALID (is_valid=true) when:\n"
            "  • The context contains concrete, on-topic information that can "
            "answer the query.\n"
            "  • Even partial coverage is acceptable if the key points are "
            "addressed.\n\n"
            "Mark INVALID (is_valid=false) when:\n"
            "  • The context is empty, completely off-topic, or only contains "
            "boilerplate / placeholder text.\n"
            "  • Critical aspects of the query are entirely unaddressed.\n\n"
            "Important: do NOT be overly strict. A good-enough answer is "
            "better than an infinite retry loop. If the context covers the "
            "core of the query, mark it VALID and let the Synthesizer refine "
            "the answer.\n\n"
            "Return ONLY valid JSON matching the required schema."
        )
    )
    user_prompt = HumanMessage(
        content=(
            f"Conversation history:\n{history_block}\n\n"
            f"Latest user query:\n{query}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            "Is this context sufficient and relevant to answer the query?"
        )
    )

    decision: CriticDecision = await critic_llm.ainvoke([system_prompt, user_prompt])

    return {
        "current_agent": "critic",
        "is_valid": decision.is_valid,
    }
