"""
Librarian node implementation.
"""

import logging

from agents.nodes.common import last_human_query
from agents.state import AgentState, RetrievedPage
from services.vector_store import query_chat_documents

log = logging.getLogger("agents.librarian")


def _fetch_pages_for_results(results: list[dict]) -> list[RetrievedPage]:
    """
    For each ChromaDB result that carries document_id + page_number metadata,
    download the source PDF from S3 (once per unique document) and slice out
    the matched page as a raw single-page PDF byte stream.

    Returns a deduplicated list of RetrievedPage dicts ordered by result rank.
    """
    from database import SessionLocal
    from models import Document
    from api.documents import _download_s3_object_to_tempfile, _slice_pdf_to_pages

    pages: list[RetrievedPage] = []
    seen: set[tuple[int, int]] = set()  # (document_id, page_number) dedup

    db = SessionLocal()
    try:
        # Cache sliced pages per document so each PDF is downloaded only once.
        page_cache: dict[int, list[bytes]] = {}

        for r in results:
            meta = r.get("metadata", {})
            raw_doc_id = meta.get("document_id")
            raw_page_num = meta.get("page_number")
            if raw_doc_id is None or raw_page_num is None:
                continue

            doc_id = int(raw_doc_id)
            page_num = int(raw_page_num)

            if (doc_id, page_num) in seen:
                continue

            if doc_id not in page_cache:
                doc = db.query(Document).filter(Document.id == doc_id).first()
                if not doc or doc.status != "ready":
                    log.warning("Librarian: doc_id=%d not ready, skipping.", doc_id)
                    page_cache[doc_id] = []
                    continue

                log.info(
                    "Librarian: downloading PDF for doc_id=%d from S3 (%s)…",
                    doc_id, doc.file_name,
                )
                temp_path = _download_s3_object_to_tempfile(doc.s3_url, doc.file_name)
                try:
                    pdf_bytes = temp_path.read_bytes()
                finally:
                    temp_path.unlink(missing_ok=True)

                page_cache[doc_id] = _slice_pdf_to_pages(pdf_bytes)
                log.info(
                    "Librarian: cached %d page(s) for doc_id=%d.",
                    len(page_cache[doc_id]), doc_id,
                )

            all_pages = page_cache[doc_id]
            page_idx = page_num - 1  # page_number is 1-based
            if 0 <= page_idx < len(all_pages):
                page_bytes = all_pages[page_idx]
                pages.append(
                    RetrievedPage(
                        document_id=doc_id,
                        page_number=page_num,
                        page_bytes=page_bytes,
                    )
                )
                seen.add((doc_id, page_num))
                log.info(
                    "Librarian: queued page %d of doc_id=%d (%d bytes) for LLM.",
                    page_num, doc_id, len(page_bytes),
                )
            else:
                log.warning(
                    "Librarian: page_number=%d out of range for doc_id=%d (total=%d).",
                    page_num, doc_id, len(all_pages),
                )
    finally:
        db.close()

    return pages


async def librarian_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks from ChromaDB scoped to the active chat,
    then fetch the matching raw PDF pages from S3 for multimodal LLM input."""
    import asyncio

    query = last_human_query(state)
    chat_id = state["chat_id"]

    log.info("Librarian querying ChromaDB | chat_id=%d | query=%.120r", chat_id, query)

    results = await asyncio.to_thread(
        query_chat_documents, query_text=query, chat_id=chat_id, n_results=5,
    )

    log.info("Librarian received %d result(s) from ChromaDB | chat_id=%d", len(results), chat_id)

    docs = []
    for i, r in enumerate(results, start=1):
        content = r.get("content", "")
        if not content:
            continue
        meta = r.get("metadata", {})
        distance = r.get("distance")
        log.info(
            "  RAG Doc %d | doc_id=%s | page=%s | chunk_type=%s | distance=%.4f | text=%.200r",
            i,
            meta.get("document_id", "?"),
            meta.get("page_number", "?"),
            meta.get("chunk_type", "?"),
            distance if distance is not None else 0.0,
            content,
        )
        docs.append(f"[Page {meta.get('page_number', '?')}] {content}")

    if not docs:
        log.info("Librarian: no uploaded documents found for chat_id=%d", chat_id)
        docs = [f"[RAG] No uploaded documents found for chat {chat_id}."]

    # Fetch actual PDF page bytes from S3 for the matched results.
    new_pages = await asyncio.to_thread(_fetch_pages_for_results, results)
    log.info(
        "Librarian: fetched %d PDF page(s) from S3 for LLM input | chat_id=%d",
        len(new_pages), chat_id,
    )

    # Deduplicate against pages already accumulated in previous retry cycles.
    existing = {
        (p["document_id"], p["page_number"])
        for p in state.get("retrieved_pages", [])
    }
    unique_new_pages = [
        p for p in new_pages
        if (p["document_id"], p["page_number"]) not in existing
    ]

    # Leave a structured signal in retrieved_context so the Critic knows that
    # real PDF pages were fetched and will be passed to the Synthesizer as
    # multimodal input.  Without this the Critic only sees short ChromaDB
    # snippets and may incorrectly judge the context as insufficient.
    page_summary: list[str] = []
    if unique_new_pages:
        page_refs = ", ".join(
            f"doc_id={p['document_id']} page {p['page_number']}"
            for p in unique_new_pages
        )
        page_summary.append(
            f"[LIBRARIAN SUMMARY] Successfully fetched {len(unique_new_pages)} "
            f"PDF page(s) for multimodal synthesis ({page_refs}). "
            "The Synthesizer will receive these pages as full PDF image content "
            "and does not depend solely on the text snippets above."
        )
        log.info(
            "Librarian: appended page summary to retrieved_context | %s",
            page_summary[0],
        )

    return {
        "current_agent": "librarian",
        "retrieved_context": state["retrieved_context"] + docs + page_summary,
        "retrieved_pages": state.get("retrieved_pages", []) + unique_new_pages,
    }
