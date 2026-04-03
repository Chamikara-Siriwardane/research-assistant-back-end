"""
Synthesizer node implementation.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.nodes.common import astream_with_retry, build_llm, last_human_query
from agents.state import AgentState


async def synthesizer_node(state: AgentState) -> dict:
    """Generate the final answer from accumulated context with token streaming enabled."""
    llm = build_llm(streaming=True)

    query = last_human_query(state)
    context_block = "\n\n".join(state["retrieved_context"]) or "No supporting context available."

    # ── Conversation history for multi-turn continuity ─────────────────────
    history_lines: list[str] = []
    for msg in state["messages"][:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content[:300]}")
    history_block = (
        "\n".join(history_lines[-6:]) if history_lines else "(no prior turns)"
    )

    system_prompt = SystemMessage(
        content=(
            "You are Jarvis, an expert research synthesizer producing PhD-caliber "
            "responses. You are given the conversation history, the user's latest "
            "query, and context retrieved by specialist agents.\n\n"
            "Guidelines:\n"
            "• Write a comprehensive, well-structured answer in Markdown.\n"
            "• Start with a brief executive summary, then provide detailed "
            "analysis organised under clear headings.\n"
            "• Reference the source labels (e.g. [RAG Doc 1], [Web Result 2]) "
            "inline as citations.\n"
            "• If the retrieved context is thin or partially relevant, still "
            "give the best answer you can and note any gaps honestly.\n"
            "• Maintain continuity with the conversation — reference prior "
            "turns when relevant rather than repeating yourself.\n"
            "• Be precise, analytical, and thorough. Avoid filler."
        )
    )
    user_prompt = HumanMessage(
        content=(
            f"Conversation history:\n{history_block}\n\n"
            f"Latest query:\n{query}\n\n"
            f"Retrieved context:\n{context_block}"
        )
    )

    full_response = ""
    async for chunk in astream_with_retry(llm, [system_prompt, user_prompt]):
        if chunk.content:
            full_response += chunk.content

    return {
        "current_agent": "synthesizer",
        "messages": [AIMessage(content=full_response)],
    }
