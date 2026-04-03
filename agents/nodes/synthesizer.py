"""
Synthesizer node implementation.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.nodes.common import build_llm, last_human_query
from agents.state import AgentState


async def synthesizer_node(state: AgentState) -> dict:
    """Generate the final answer from accumulated context with token streaming enabled."""
    llm = build_llm(streaming=True)

    query = last_human_query(state)
    context_block = "\n\n".join(state["retrieved_context"]) or "No supporting context available."

    system_prompt = SystemMessage(
        content=(
            "You are an expert research synthesizer writing PhD-caliber responses. "
            "Using only the provided context, write a comprehensive, well-structured "
            "answer in Markdown. Include:\n"
            "  - A brief executive summary\n"
            "  - Detailed analysis organised under headings\n"
            "  - Inline citations referencing the source labels provided\n"
            "Be precise, analytical, and thorough."
        )
    )
    user_prompt = HumanMessage(
        content=(
            f"Research question:\n{query}\n\n"
            f"Retrieved context:\n{context_block}"
        )
    )

    full_response = ""
    async for chunk in llm.astream([system_prompt, user_prompt]):
        if chunk.content:
            full_response += chunk.content

    return {
        "current_agent": "synthesizer",
        "messages": [AIMessage(content=full_response)],
    }
