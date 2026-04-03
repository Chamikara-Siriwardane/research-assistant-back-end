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

    system_prompt = SystemMessage(
        content=(
            "You are a rigorous quality-control critic for a research assistant. "
            "Evaluate whether the retrieved context is sufficient and directly relevant "
            "to answer the user's query in full. Be strict:\n"
            "  - Mark INVALID if the context is vague, empty, tangential, or incomplete.\n"
            "  - Mark VALID only when the context provides clear, on-topic information.\n"
            "Return ONLY valid JSON matching the required schema."
        )
    )
    user_prompt = HumanMessage(
        content=(
            f"User query:\n{query}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            "Is this context sufficient and relevant?"
        )
    )

    decision: CriticDecision = await critic_llm.ainvoke([system_prompt, user_prompt])

    return {
        "current_agent": "critic",
        "is_valid": decision.is_valid,
    }
