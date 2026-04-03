"""
Analyst node implementation.
"""

import asyncio

from agents.state import AgentState


async def analyst_node(state: AgentState) -> dict:
    """Simulate code execution and append execution output."""
    await asyncio.sleep(0.6)
    mock_output = (
        "[Code Execution Result]\n"
        ">>> import statistics\n"
        ">>> data = [2.1, 3.7, 1.9, 4.5, 3.2]\n"
        ">>> statistics.mean(data)\n"
        "3.08\n"
        ">>> statistics.stdev(data)\n"
        "0.978  (mock values - replace with real REPL output)"
    )

    return {
        "current_agent": "analyst",
        "retrieved_context": state["retrieved_context"] + [mock_output],
    }
