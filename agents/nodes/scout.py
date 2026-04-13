"""
agents/nodes/scout.py
---------------------
Scout node — Deterministic tool orchestration for web and academic search.

Tools
-----
  search_academic_papers   : Google Scholar via SerpAPI
  general_web_search       : Standard Google search via SerpAPI
  download_and_extract_pdf : Download a PDF and extract its text via PyMuPDF

Architecture
------------
The Gemini 2.5 model series runs in thinking mode and produces
AIMessage.content='' on every turn, which breaks LangGraph's create_react_agent
loop (the model's reasoning is invisible to LangChain's interface).

To work around this, Scout uses a **deterministic Python orchestration loop**
instead of a ReAct agent.  The tool-call sequence is controlled entirely by
code; the LLM is invoked exactly once at the end to synthesise a final answer
from the gathered evidence.  This produces reliable, visible output regardless
of the underlying model's thinking mode.

Orchestration flow
------------------
  1. search_academic_papers
  2. If title match found AND PDF available → download_and_extract_pdf
  3. If NO title match → general_web_search
     a. If PDF/arXiv link found in web results → download_and_extract_pdf
  4. LLM synthesis call (single invocation, always produces text output)
"""

from __future__ import annotations

import io
import logging
import re
import textwrap
from typing import Any

import fitz  # PyMuPDF
import requests
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agents.nodes.common import ainvoke_with_retry, build_llm, last_human_query
from agents.state import AgentState
from core.config import settings

log = logging.getLogger("agents.scout")

# ---------------------------------------------------------------------------
# SerpAPI key helper
# ---------------------------------------------------------------------------

def _serpapi_key() -> str:
    """Return the SerpAPI key from config / environment."""
    key = settings.serp_api_key
    if not key:
        raise RuntimeError(
            "SerpAPI key not found. Set SERP_API_KEY in your .env file."
        )
    return key


# ---------------------------------------------------------------------------
# Tool 1 — Academic paper search (Google Scholar)
# ---------------------------------------------------------------------------

@tool
def search_academic_papers(query: str) -> str:
    """Search Google Scholar for academic papers matching the given query.

    Handles fuzzy or partial hints (e.g., "MoE paper from last year").
    Returns a structured list of: title, authors, snippet, and a direct PDF
    link when SerpAPI exposes one in the resources block.

    Args:
        query: Natural-language search hint or paper title fragment.

    Returns:
        A formatted string listing found papers with their metadata.
    """
    log.info("[Scout/Tool] search_academic_papers › query=%r", query)

    try:
        wrapper = SerpAPIWrapper(
            serpapi_api_key=_serpapi_key(),
            params={"engine": "google_scholar"},
        )
        log.debug("[Scout/Tool] search_academic_papers › calling SerpAPI…")
        raw: dict[str, Any] = wrapper.results(query)
    except Exception as exc:
        log.error("[Scout/Tool] search_academic_papers › SerpAPI call failed: %s", exc)
        return f"Error querying Google Scholar: {exc}"

    organic: list[dict] = raw.get("organic_results", [])
    if not organic:
        log.warning("[Scout/Tool] search_academic_papers › zero results returned")
        return "No academic results found for this query."

    log.info(
        "[Scout/Tool] search_academic_papers › %d results received, parsing…",
        len(organic),
    )

    lines: list[str] = []
    for idx, paper in enumerate(organic[:8], start=1):
        title: str = paper.get("title", "Unknown title")
        snippet: str = paper.get("snippet", "No snippet available.")

        # Authors / year live inside publication_info
        pub_info: dict = paper.get("publication_info", {})
        raw_authors: list[dict] | None = pub_info.get("authors")
        if raw_authors:
            author_str = ", ".join(a.get("name", "") for a in raw_authors[:3])
            if len(raw_authors) > 3:
                author_str += " et al."
        else:
            author_str = pub_info.get("summary", "Unknown authors")

        # Extract the first PDF link from the resources block, if present
        pdf_link = ""
        for res in paper.get("resources", []):
            if str(res.get("file_format", "")).upper() == "PDF":
                pdf_link = res.get("link", "")
                break

        log.debug(
            "[Scout/Tool] search_academic_papers › [%d] %r — pdf=%r",
            idx, title, pdf_link or "none",
        )

        lines.append(f"[{idx}] {title}")
        lines.append(f"    Authors : {author_str}")
        lines.append(f"    Snippet : {snippet[:300]}")
        if pdf_link:
            lines.append(f"    PDF     : {pdf_link}")
        lines.append("")

    result = "\n".join(lines)
    log.info(
        "[Scout/Tool] search_academic_papers › formatted output (%d chars)", len(result)
    )
    return result


# ---------------------------------------------------------------------------
# Tool 2 — General web search
# ---------------------------------------------------------------------------

@tool
def general_web_search(query: str) -> str:
    """Search the general web for non-academic or current-events queries.

    Args:
        query: The search query string.

    Returns:
        A formatted string of top results with titles, snippets, and URLs.
    """
    log.info("[Scout/Tool] general_web_search › query=%r", query)

    try:
        wrapper = SerpAPIWrapper(
            serpapi_api_key=_serpapi_key(),
            params={"engine": "google", "num": "10"},
        )
        log.debug("[Scout/Tool] general_web_search › calling SerpAPI…")
        raw: dict[str, Any] = wrapper.results(query)
    except Exception as exc:
        log.error("[Scout/Tool] general_web_search › SerpAPI call failed: %s", exc)
        return f"Error performing web search: {exc}"

    organic: list[dict] = raw.get("organic_results", [])
    if not organic:
        log.warning("[Scout/Tool] general_web_search › zero results returned")
        return "No web results found for this query."

    log.info(
        "[Scout/Tool] general_web_search › %d results received, parsing…",
        len(organic),
    )

    lines: list[str] = []
    for idx, result in enumerate(organic[:6], start=1):
        title: str = result.get("title", "No title")
        snippet: str = result.get("snippet", "No snippet available.")
        link: str = result.get("link", "")

        log.debug("[Scout/Tool] general_web_search › [%d] %r", idx, title)

        lines.append(f"[{idx}] {title}")
        lines.append(f"    URL     : {link}")
        lines.append(f"    Snippet : {snippet[:350]}")
        lines.append("")

    result_str = "\n".join(lines)
    log.info(
        "[Scout/Tool] general_web_search › formatted output (%d chars)", len(result_str)
    )
    return result_str


# ---------------------------------------------------------------------------
# Tool 3 — PDF downloader + text extractor
# ---------------------------------------------------------------------------

@tool
def download_and_extract_pdf(url: str) -> str:
    """Download a PDF from the given direct URL and extract its full text.

    Uses ``requests`` for the HTTP download and ``PyMuPDF`` (fitz) for
    text parsing.  Handles paywalled or inaccessible PDFs gracefully.

    Args:
        url: A direct HTTPS link pointing to a PDF file.

    Returns:
        Extracted plain text from the PDF (up to 15 000 characters), or a
        descriptive error message if the download or parse fails.
    """
    log.info("[Scout/Tool] download_and_extract_pdf › url=%s", url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*;q=0.9",
    }

    # ── HTTP download ──────────────────────────────────────────────────────
    try:
        log.debug("[Scout/Tool] download_and_extract_pdf › sending GET request…")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        log.info(
            "[Scout/Tool] download_and_extract_pdf › downloaded %d bytes (status %d)",
            len(response.content),
            response.status_code,
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        log.warning(
            "[Scout/Tool] download_and_extract_pdf › HTTP %s for %s", status, url
        )
        if status in (401, 403):
            return (
                f"Access denied (HTTP {status}). This PDF may be paywalled or "
                "require authentication. Try finding an open-access mirror."
            )
        return f"HTTP error {status} while downloading PDF: {exc}"
    except requests.exceptions.Timeout:
        log.error("[Scout/Tool] download_and_extract_pdf › request timed out")
        return "Request timed out while downloading the PDF. The server may be slow."
    except requests.exceptions.RequestException as exc:
        log.error("[Scout/Tool] download_and_extract_pdf › request error: %s", exc)
        return f"Network error while downloading PDF: {exc}"

    # Sanity-check the content type
    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not url.lower().split("?")[0].endswith(".pdf"):
        log.warning(
            "[Scout/Tool] download_and_extract_pdf › unexpected Content-Type=%r",
            content_type,
        )

    # ── PDF parsing ─────────────────────────────────────────────────────────
    try:
        log.debug("[Scout/Tool] download_and_extract_pdf › opening PDF with fitz…")
        doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
    except Exception as exc:
        log.error(
            "[Scout/Tool] download_and_extract_pdf › fitz parsing failed: %s", exc
        )
        return f"Failed to parse the downloaded content as a PDF: {exc}"

    total_pages = len(doc)
    log.info(
        "[Scout/Tool] download_and_extract_pdf › PDF opened — %d pages", total_pages
    )

    text_parts: list[str] = []
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

    doc.close()

    if not text_parts:
        log.warning(
            "[Scout/Tool] download_and_extract_pdf › no extractable text in PDF "
            "(may be scanned / image-only)"
        )
        return (
            "The PDF appears to contain no extractable text (possibly a scanned "
            "image-only document). Consider searching for an HTML or text version."
        )

    full_text = "\n\n".join(text_parts)
    _CHAR_LIMIT = 15_000
    if len(full_text) > _CHAR_LIMIT:
        truncated = full_text[:_CHAR_LIMIT] + "\n\n[... text truncated for context window ...]"
        log.info(
            "[Scout/Tool] download_and_extract_pdf › truncated %d → %d chars",
            len(full_text),
            _CHAR_LIMIT,
        )
    else:
        truncated = full_text

    log.info(
        "[Scout/Tool] download_and_extract_pdf › extraction complete (%d chars returned)",
        len(truncated),
    )
    return truncated


# ---------------------------------------------------------------------------
# LLM relevance check — replaces fragile token-overlap heuristic
# ---------------------------------------------------------------------------

class _ScholarRelevance(BaseModel):
    """Structured output for the Scholar relevance decision."""
    has_match: bool = Field(
        description=(
            "True if at least one result is clearly about the paper or topic "
            "the user asked for. False if all results are only tangentially related."
        )
    )
    best_pdf_url: str = Field(
        default="",
        description=(
            "Direct PDF URL of the best matching result, if available. "
            "Empty string if no match or no PDF."
        ),
    )
    reasoning: str = Field(description="One-sentence explanation of the decision.")


async def _llm_check_scholar_relevance(
    query: str, scholar_text: str
) -> _ScholarRelevance:
    """Ask the LLM whether Scholar results satisfy the user's query.

    This is a cheap, fast structured-output call that replaces the brittle
    token-overlap heuristic.  It correctly handles conversational queries,
    fuzzy paper names, and partial matches that token overlap would miss.

    Args:
        query:        The original user query.
        scholar_text: Formatted Scholar results string.

    Returns:
        A :class:`_ScholarRelevance` instance with has_match, best_pdf_url,
        and reasoning.
    """
    llm = build_llm()
    structured_llm = llm.with_structured_output(_ScholarRelevance)

    system = SystemMessage(
        content=(
            "You are a research assistant evaluating whether a set of Google "
            "Scholar search results satisfactorily covers the user's query.\n\n"
            "Decide has_match=True if at least one result is clearly about the "
            "paper or topic the user named or described — even if the title "
            "wording differs slightly.\n"
            "Decide has_match=False if no result is about what the user asked "
            "for (e.g. results are on a completely different topic).\n"
            "If has_match=True, also provide the PDF URL of the best result "
            "(prefer arXiv or open-access links). If no PDF is listed, leave "
            "best_pdf_url empty.\n"
            "Return ONLY valid JSON matching the required schema."
        )
    )
    user = HumanMessage(
        content=(
            f"User query:\n{query}\n\n"
            f"Scholar results:\n{scholar_text}"
        )
    )
    try:
        result: _ScholarRelevance = await ainvoke_with_retry(
            structured_llm, [system, user]
        )
        return result
    except Exception as exc:
        log.warning(
            "[Scout] LLM relevance check failed (%s) — defaulting to no match.", exc
        )
        return _ScholarRelevance(has_match=False, best_pdf_url="", reasoning=str(exc))


# Tokens that strongly suggest a URL points to a PDF or arXiv abstract
_PDF_URL_RE = re.compile(r"arxiv\.org/(abs|pdf)/|\.pdf(\?|$)", re.IGNORECASE)


def _extract_pdf_link_from_scholar(raw_results: dict) -> str:
    """Extract the first PDF URL from SerpAPI Scholar organic results."""
    for paper in raw_results.get("organic_results", []):
        for res in paper.get("resources", []):
            if str(res.get("file_format", "")).upper() == "PDF":
                link = res.get("link", "")
                if link:
                    return link
    return ""


def _extract_pdf_link_from_web(raw_results: dict) -> str:
    """Extract the first arXiv/PDF link from SerpAPI web organic results."""
    for item in raw_results.get("organic_results", []):
        link = item.get("link", "")
        if link and _PDF_URL_RE.search(link):
            # Prefer /pdf/ over /abs/ so fitz can parse directly
            pdf_link = link.replace("/abs/", "/pdf/")
            log.debug("[Scout] _extract_pdf_link_from_web › found %s", pdf_link)
            return pdf_link
    return ""


def _format_scholar_results(raw: dict) -> str:
    """Format raw SerpAPI Scholar results into a readable string."""
    lines: list[str] = []
    for idx, paper in enumerate(raw.get("organic_results", [])[:8], start=1):
        title = paper.get("title", "Unknown title")
        snippet = paper.get("snippet", "")
        pub_info = paper.get("publication_info", {})
        raw_authors = pub_info.get("authors")
        if raw_authors:
            author_str = ", ".join(a.get("name", "") for a in raw_authors[:3])
            if len(raw_authors) > 3:
                author_str += " et al."
        else:
            author_str = pub_info.get("summary", "Unknown authors")
        pdf_link = ""
        for res in paper.get("resources", []):
            if str(res.get("file_format", "")).upper() == "PDF":
                pdf_link = res.get("link", "")
                break
        lines.append(f"[{idx}] {title}")
        lines.append(f"    Authors : {author_str}")
        if snippet:
            lines.append(f"    Snippet : {snippet[:300]}")
        if pdf_link:
            lines.append(f"    PDF     : {pdf_link}")
        lines.append("")
    return "\n".join(lines) if lines else "No results."


def _format_web_results(raw: dict) -> str:
    """Format raw SerpAPI web results into a readable string."""
    lines: list[str] = []
    for idx, item in enumerate(raw.get("organic_results", [])[:6], start=1):
        title = item.get("title", "No title")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        lines.append(f"[{idx}] {title}")
        lines.append(f"    URL     : {link}")
        if snippet:
            lines.append(f"    Snippet : {snippet[:350]}")
        lines.append("")
    return "\n".join(lines) if lines else "No results."


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = textwrap.dedent("""
    You are Scout, a research specialist inside Jarvis, a PhD-calibre AI
    research assistant.

    You have just gathered the following evidence from web and academic
    searches (and optionally a full PDF read).  Your task is to write a
    comprehensive, well-structured answer to the user's query based solely
    on this evidence.

    Include:
      • Key findings / claims with inline source citations (title or URL).
      • Authors and publication year for academic works.
      • Direct quotes or paraphrases from any PDF text that was retrieved.
      • If the exact paper was not found, clearly state that and summarise
        the closest related work.

    Be thorough but concise.
""").strip()


# ---------------------------------------------------------------------------
# Scout node — deterministic orchestration
# ---------------------------------------------------------------------------

async def scout_node(state: AgentState) -> dict:
    """LangGraph node: deterministic search orchestration + LLM synthesis.

    Replaces the create_react_agent approach, which is broken with Gemini 2.5
    thinking models (AIMessage.content is always '' in thinking mode, making
    the ReAct loop invisible and non-functional).

    Orchestration order
    -------------------
    1. search_academic_papers (Google Scholar)
    2a. Title match found + PDF available → download_and_extract_pdf
    2b. No title match          → general_web_search (Google)
        ↳ arXiv/PDF link found  → download_and_extract_pdf
    3. Single LLM synthesis call (always produces visible text)

    Args:
        state: Current ``AgentState`` from the main LangGraph workflow.

    Returns:
        Partial state update with ``current_agent`` and ``retrieved_context``.
    """
    log.info("[Scout] ══════════════ Scout node invoked ══════════════")

    query = last_human_query(state)
    if not query:
        log.warning("[Scout] No human query found in state — returning empty context.")
        return {
            "current_agent": "scout",
            "retrieved_context": state["retrieved_context"],
        }

    log.info("[Scout] Query: %r", query)
    log.info("[Scout] Existing context items: %d", len(state["retrieved_context"]))

    evidence_blocks: list[str] = []
    serpapi_key = _serpapi_key()

    # ── Step 1: Google Scholar ──────────────────────────────────────────────
    log.info("[Scout] Step 1 › search_academic_papers › query=%r", query)
    try:
        scholar_wrapper = SerpAPIWrapper(
            serpapi_api_key=serpapi_key,
            params={"engine": "google_scholar"},
        )
        scholar_raw: dict[str, Any] = scholar_wrapper.results(query)
        scholar_organic = scholar_raw.get("organic_results", [])
        log.info("[Scout] Step 1 › Scholar returned %d results.", len(scholar_organic))
    except Exception as exc:
        log.error("[Scout] Step 1 › Scholar search failed: %s", exc)
        scholar_raw = {}
        scholar_organic = []

    scholar_text = _format_scholar_results(scholar_raw)
    if scholar_organic:
        evidence_blocks.append(f"[Google Scholar results]\n{scholar_text}")

    # ── Step 2: LLM relevance check on Scholar results ─────────────────────
    log.info("[Scout] Step 2 › asking LLM whether Scholar results are relevant…")
    relevance = await _llm_check_scholar_relevance(query, scholar_text)
    log.info(
        "[Scout] Step 2 › has_match=%s | best_pdf=%r | reasoning=%r",
        relevance.has_match,
        relevance.best_pdf_url,
        relevance.reasoning,
    )

    best_pdf_link: str = ""

    if relevance.has_match:
        best_pdf_link = relevance.best_pdf_url or _extract_pdf_link_from_scholar(scholar_raw)
        log.info("[Scout] Step 2a › Scholar match confirmed. PDF link: %s", best_pdf_link or "none")
    else:
        # ── Step 2b: fall back to general web search ────────────────────────
        log.info("[Scout] Step 2b › No Scholar match — falling back to general_web_search.")
        try:
            web_wrapper = SerpAPIWrapper(
                serpapi_api_key=serpapi_key,
                params={"engine": "google", "num": "10"},
            )
            web_raw: dict[str, Any] = web_wrapper.results(query)
            web_organic = web_raw.get("organic_results", [])
            log.info("[Scout] Step 2b › Web search returned %d results.", len(web_organic))
        except Exception as exc:
            log.error("[Scout] Step 2b › Web search failed: %s", exc)
            web_raw = {}
            web_organic = []

        web_text = _format_web_results(web_raw)
        if web_organic:
            evidence_blocks.append(f"[General web search results]\n{web_text}")

        best_pdf_link = _extract_pdf_link_from_web(web_raw)
        log.info("[Scout] Step 2b › PDF from web: %s", best_pdf_link or "none")

    # ── Step 3: download PDF if a link was found ────────────────────────────
    if best_pdf_link:
        log.info("[Scout] Step 3 › download_and_extract_pdf › url=%s", best_pdf_link)
        pdf_text = download_and_extract_pdf.invoke({"url": best_pdf_link})  # type: ignore[arg-type]
        evidence_blocks.append(f"[PDF content from {best_pdf_link}]\n{pdf_text}")
    else:
        log.info("[Scout] Step 3 › No PDF link available — skipping download.")

    if not evidence_blocks:
        log.warning("[Scout] No evidence gathered — returning empty context.")
        return {
            "current_agent": "scout",
            "retrieved_context": state["retrieved_context"]
            + ["[SCOUT FINDINGS]\nNo results found for this query."],
        }

    # ── Step 4: LLM synthesis (single call, always produces visible text) ───
    log.info(
        "[Scout] Step 4 › LLM synthesis | evidence_blocks=%d", len(evidence_blocks)
    )
    evidence_body = "\n\n".join(evidence_blocks)
    llm = build_llm()
    synthesis_prompt = [
        SystemMessage(content=_SYNTHESIS_SYSTEM),
        HumanMessage(
            content=(
                f"User query:\n{query}\n\n"
                f"Evidence gathered:\n{evidence_body}"
            )
        ),
    ]
    try:
        synthesis_response = await ainvoke_with_retry(llm, synthesis_prompt)
        final_answer = synthesis_response.content
        if isinstance(final_answer, list):
            final_answer = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in final_answer
                if not (isinstance(p, dict) and p.get("type") == "thinking")
            )
        final_answer = str(final_answer).strip()
        log.info("[Scout] Step 4 › Synthesis complete (%d chars).", len(final_answer))
    except Exception as exc:
        log.error("[Scout] Step 4 › LLM synthesis failed: %s", exc, exc_info=True)
        # Fall back to raw evidence so the Critic has something to evaluate
        final_answer = evidence_body

    context_entry = f"[SCOUT FINDINGS]\n{final_answer}"
    log.info("[Scout] ══════════════ Scout node complete ══════════════")
    return {
        "current_agent": "scout",
        "retrieved_context": state["retrieved_context"] + [context_entry],
    }
