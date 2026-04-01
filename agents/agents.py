"""
agents/agents.py
----------------
LangGraph node implementations for every agent in the research pipeline.

All LLM calls use ChatGoogleGenerativeAI (Gemini).
External tool calls (vector DB, web search, code execution) are **mocked**
with asyncio.sleep stubs — replace them with real tool implementations
without changing any node signature.

Node contract
-------------
  async (state: AgentState) -> dict   — returns a *partial* state update
"""

import asyncio
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from agents.state import AgentState, RouteCommand
from core.config import settings


# ---------------------------------------------------------------------------
# Shared LLM factory
# ---------------------------------------------------------------------------

def _build_llm(streaming: bool = False) -> ChatGoogleGenerativeAI:
    """Construct a ChatGoogleGenerativeAI instance with project-wide defaults."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Internal structured-output schemas (never exposed as SSE events)
# ---------------------------------------------------------------------------

class _RouterDecision(BaseModel):
    """Output schema for the Supervisor's routing decision."""

    route: RouteCommand = Field(
        description=(
            "Which specialist agent to invoke next.  Choose 'route_to_synthesizer' "
            "only when the retrieved_context is already sufficient."
        )
    )
    reasoning: str = Field(description="One-sentence justification for the routing choice.")


class _CriticDecision(BaseModel):
    """Output schema for the Critic's validation verdict."""

    is_valid: bool = Field(
        description=(
            "True if the retrieved context is relevant and sufficient to answer "
            "the query in full.  False if it is vague, off-topic, or incomplete."
        )
    )
    reasoning: str = Field(description="Brief explanation of the verdict.")


# ---------------------------------------------------------------------------
# Helper — extract the most recent human message from state
# ---------------------------------------------------------------------------

def _last_human_query(state: AgentState) -> str:
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )


# ---------------------------------------------------------------------------
# 1. Supervisor — analyses the query and decides which specialist to call
# ---------------------------------------------------------------------------

async def supervisor_node(state: AgentState) -> dict:
    """
    Reads the latest human message and decides routing strategy.

    Uses `with_structured_output` to guarantee a typed _RouterDecision
    response, which is then mapped to a RouteCommand stored in state.

    Returns: route_command, current_agent
    """
    llm = _build_llm()
    router_llm = llm.with_structured_output(_RouterDecision)

    query = _last_human_query(state)

    system_prompt = SystemMessage(
        content=(
            "You are the supervisor of a PhD-caliber research assistant. "
            "Analyse the user's query and decide which specialist agent to invoke:\n\n"
            "  route_to_rag        — query needs information from a private document store\n"
            "  route_to_web        — query needs recent, real-time, or news-based information\n"
            "  route_to_code       — query needs data analysis, calculations, or code execution\n"
            "  route_to_synthesizer — sufficient context is already in state; skip retrieval\n\n"
            "Return ONLY valid JSON matching the required schema."
        )
    )

    decision: _RouterDecision = await router_llm.ainvoke(
        [system_prompt, HumanMessage(content=query)]
    )

    return {
        "current_agent": "supervisor",
        "route_command": decision.route,
    }


# ---------------------------------------------------------------------------
# 2. Librarian — local RAG agent (mocked vector database search)
# ---------------------------------------------------------------------------

async def librarian_node(state: AgentState) -> dict:
    """
    Simulates a semantic search against a private vector database.

    MOCK: Replace the body below with a real retriever call, e.g.:
        retriever = Chroma(...).as_retriever()
        docs = await retriever.ainvoke(query)

    Returns: retrieved_context (appended), current_agent
    """
    # ── MOCK ─────────────────────────────────────────────────────────────
    await asyncio.sleep(0.5)  # simulate I/O latency
    query = _last_human_query(state)
    mock_docs = [
        f"[RAG Doc 1] Relevant passage about '{query}' retrieved from the internal knowledge base.",
        "[RAG Doc 2] Supporting evidence from a stored research paper (mock).",
    ]
    # ── END MOCK ──────────────────────────────────────────────────────────

    return {
        "current_agent": "librarian",
        "retrieved_context": state["retrieved_context"] + mock_docs,
    }


# ---------------------------------------------------------------------------
# 3. Scout — web search agent (mocked Tavily / SerpAPI call)
# ---------------------------------------------------------------------------

async def scout_node(state: AgentState) -> dict:
    """
    Simulates an external web search (e.g. Tavily, SerpAPI, Bing Search).

    MOCK: Replace the body below with a real tool call, e.g.:
        tool = TavilySearchResults(max_results=5)
        results = await tool.ainvoke(query)

    Returns: retrieved_context (appended), current_agent
    """
    # ── MOCK ─────────────────────────────────────────────────────────────
    await asyncio.sleep(0.8)  # simulate network latency
    query = _last_human_query(state)
    mock_results = [
        f"[Web Result 1] Breaking research on '{query}' — arxiv.org/abs/mock-2026.",
        "[Web Result 2] Recent survey article covering key findings (mock).",
        "[Web Result 3] Official documentation snippet (mock).",
    ]
    # ── END MOCK ──────────────────────────────────────────────────────────

    return {
        "current_agent": "scout",
        "retrieved_context": state["retrieved_context"] + mock_results,
    }


# ---------------------------------------------------------------------------
# 4. Analyst — code execution agent (mocked Python REPL)
# ---------------------------------------------------------------------------

async def analyst_node(state: AgentState) -> dict:
    """
    Simulates a sandboxed Python REPL for data analysis and calculations.

    MOCK: Replace the body below with a real code execution tool, e.g.:
        tool = PythonREPLTool()
        output = await tool.ainvoke(generated_code)

    Returns: retrieved_context (appended), current_agent
    """
    # ── MOCK ─────────────────────────────────────────────────────────────
    await asyncio.sleep(0.6)  # simulate execution time
    mock_output = (
        "[Code Execution Result]\n"
        ">>> import statistics\n"
        ">>> data = [2.1, 3.7, 1.9, 4.5, 3.2]\n"
        ">>> statistics.mean(data)\n"
        "3.08\n"
        ">>> statistics.stdev(data)\n"
        "0.978  (mock values — replace with real REPL output)"
    )
    # ── END MOCK ──────────────────────────────────────────────────────────

    return {
        "current_agent": "analyst",
        "retrieved_context": state["retrieved_context"] + [mock_output],
    }


# ---------------------------------------------------------------------------
# 5. Critic — validates retrieved context against the original query
# ---------------------------------------------------------------------------

async def critic_node(state: AgentState) -> dict:
    """
    Reviews the accumulated `retrieved_context` and decides whether it
    adequately answers the user's query.

    - is_valid = True  → graph routes to Synthesizer.
    - is_valid = False → graph loops back to Supervisor for a retry.

    Uses `with_structured_output` for a guaranteed typed verdict.

    Returns: is_valid, current_agent
    """
    llm = _build_llm()
    critic_llm = llm.with_structured_output(_CriticDecision)

    query = _last_human_query(state)
    context_block = "\n".join(state["retrieved_context"]) or "No context retrieved yet."

    system_prompt = SystemMessage(
        content=(
            "You are a rigorous quality-control critic for a research assistant. "
            "Evaluate whether the retrieved context is *sufficient and directly relevant* "
            "to answer the user's query in full.  Be strict:\n"
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

    decision: _CriticDecision = await critic_llm.ainvoke([system_prompt, user_prompt])

    return {
        "current_agent": "critic",
        "is_valid": decision.is_valid,
    }


# ---------------------------------------------------------------------------
# 6. Synthesizer — generates the final streamed markdown response
# ---------------------------------------------------------------------------

async def synthesizer_node(state: AgentState) -> dict:
    """
    Combines the conversation history and all retrieved context to produce
    the final, structured markdown answer.

    Uses `llm.astream()` with `streaming=True` so that every generated token
    surfaces as an `on_chat_model_stream` event when the graph is consumed
    via `compiled_graph.astream_events()`.  The orchestrator layer listens
    for those events and forwards them to the SSE stream.

    Returns: messages (final AIMessage appended), current_agent
    """
    # streaming=True so tokens bubble up through astream_events
    llm = _build_llm(streaming=True)

    query = _last_human_query(state)
    context_block = "\n\n".join(state["retrieved_context"]) or "No supporting context available."

    system_prompt = SystemMessage(
        content=(
            "You are an expert research synthesizer writing PhD-caliber responses. "
            "Using only the provided context, write a comprehensive, well-structured "
            "answer in Markdown.  Include:\n"
            "  • A brief executive summary\n"
            "  • Detailed analysis organised under headings\n"
            "  • Inline citations referencing the source labels provided\n"
            "Be precise, analytical, and thorough."
        )
    )
    user_prompt = HumanMessage(
        content=(
            f"Research question:\n{query}\n\n"
            f"Retrieved context:\n{context_block}"
        )
    )

    # Stream tokens — each chunk fires an `on_chat_model_stream` event
    # that the orchestrator translates into a TextEvent SSE payload.
    full_response = ""
    async for chunk in llm.astream([system_prompt, user_prompt]):
        if chunk.content:
            full_response += chunk.content

    return {
        "current_agent": "synthesizer",
        "messages": [AIMessage(content=full_response)],
    }
