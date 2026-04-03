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


async def supervisor_node(state: AgentState) -> dict:
    """Analyze the latest query and route to the next specialist node."""
    llm = build_llm()
    router_llm = llm.with_structured_output(RouterDecision)

    query = last_human_query(state)

    system_prompt = SystemMessage(
        content=(
            "You are the supervisor of a PhD-caliber research assistant. "
            "Analyse the user's query and decide which specialist agent to invoke:\n\n"
            "  route_to_rag - query needs information from a private document store\n"
            "  route_to_web - query needs recent, real-time, or news-based information\n"
            "  route_to_code - query needs data analysis, calculations, or code execution\n"
            "  route_to_synthesizer - sufficient context is already in state; skip retrieval\n\n"
            "Return ONLY valid JSON matching the required schema."
        )
    )

    decision: RouterDecision = await router_llm.ainvoke(
        [system_prompt, HumanMessage(content=query)]
    )

    return {
        "current_agent": "supervisor",
        "route_command": decision.route,
    }
