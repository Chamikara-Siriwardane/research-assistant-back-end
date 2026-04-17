"""
services/vector_store.py
-----------------------
Global ChromaDB service for document chunk storage and chat-scoped retrieval.

Architecture
------------
Uses a single persistent Chroma collection named ``jarvis_global_knowledge``
stored under ``./chroma_data``. All chunks are tagged with ``chat_id`` and
``document_id`` metadata so queries can be filtered to the active chat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from core.config import settings

COLLECTION_NAME = "jarvis_global_knowledge"
PERSIST_DIRECTORY = "./chroma_data"

_client: chromadb.PersistentClient | None = None
_collection: Any | None = None

# Lazy-loaded Gemini text embeddings.
# gemini-embedding-2-preview does not accept a task_type enum parameter for
# text inputs. Asymmetric retrieval is achieved instead by embedding text
# prefixes directly into the input content (see helpers below).
_embeddings: GoogleGenerativeAIEmbeddings | None = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the shared Gemini text embeddings client."""
    global _embeddings
    if _embeddings is None:
        api_key = SecretStr(settings.gemini_api_key) if settings.gemini_api_key else None
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=api_key,
        )
    return _embeddings


def _format_document_for_embedding(text: str, title: str | None = None) -> str:
    """Apply the gemini-embedding-2-preview document prefix format.

    Format: ``title: {title} | text: {content}``
    Per the official docs, use ``title: none`` when no title is available.
    """
    return f"title: {title or 'none'} | text: {text}"


def _format_query_for_embedding(query: str) -> str:
    """Apply the gemini-embedding-2-preview search-query prefix format.

    Format: ``task: search result | query: {content}``
    """
    return f"task: search result | query: {query}"


class _GeminiEmbeddingFunction:
    """Wrap Gemini embeddings for Chroma's embedding_function interface."""

    def name(self) -> str:
        return settings.embedding_model

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed documents using the gemini-embedding-2-preview document prefix format."""
        formatted = [_format_document_for_embedding(doc) for doc in input]
        return _get_embeddings().embed_documents(formatted)


def _get_client() -> chromadb.PersistentClient:
    """Return the shared persistent Chroma client."""
    global _client
    if _client is None:
        Path(PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    return _client


def get_collection() -> Any:
    """Create or return the shared global collection."""
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_GeminiEmbeddingFunction(),
        )
    return _collection


def add_multimodal_pdf_pages(
    embeddings: list[list[float]],
    chat_id: int,
    document_id: int,
) -> list[str]:
    """Insert pre-computed multimodal PDF page embeddings into the global collection.

    Each page gets its own vector entry with the following metadata:
    - ``chat_id``       — scopes the vector to the originating chat
    - ``document_id``   — links back to the source Document row
    - ``page_number``   — 1-based page index (enables fast page-targeted retrieval)
    - ``chunk_type``    — always ``"multimodal_pdf_page"``

    Embeddings are passed directly to ChromaDB so the collection's internal
    embedding function is bypassed entirely.
    """
    if not embeddings:
        return []

    page_count = len(embeddings)
    ids = [f"doc_{document_id}_page_{page_num}" for page_num in range(1, page_count + 1)]
    documents = [f"PDF page {page_num}" for page_num in range(1, page_count + 1)]
    metadatas = [
        {
            "chat_id": chat_id,
            "document_id": document_id,
            "page_number": page_num,
            "chunk_type": "multimodal_pdf_page",
        }
        for page_num in range(1, page_count + 1)
    ]

    get_collection().add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    return ids


def add_document_chunks(
    chunks: list[str],
    chat_id: int,
    document_id: int,
    extra_metadatas: list[dict] | None = None,
) -> list[str]:
    """Insert document chunks into the global collection with per-chunk metadata.

    *extra_metadatas*, when provided, must have the same length as *chunks*.
    Each entry is merged with the base ``chat_id`` / ``document_id`` metadata so
    callers can attach page numbers, header breadcrumbs, etc.
    All metadata values must be scalars (str, int, float, or bool) to satisfy
    ChromaDB's constraints.
    """
    if not chunks:
        return []

    ids = [f"doc_{document_id}_chunk_{index}" for index, _ in enumerate(chunks, start=1)]
    base = {"chat_id": chat_id, "document_id": document_id}
    if extra_metadatas:
        metadatas = [
            {**base, **{k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool))}}
            for extra in extra_metadatas
        ]
    else:
        metadatas = [base for _ in chunks]

    get_collection().add(documents=chunks, metadatas=metadatas, ids=ids)
    return ids


def query_chat_documents(query_text: str, chat_id: int, n_results: int = 5) -> list[dict[str, Any]]:
    """Query the global collection, restricted to the active chat via metadata filter."""
    if not query_text.strip():
        return []

    # Apply gemini-embedding-2-preview search-query prefix format, then pass
    # the vector directly to ChromaDB so the collection's embedding function
    # (document-side) is not invoked for queries.
    formatted_query = _format_query_for_embedding(query_text)
    query_vector = _get_embeddings().embed_query(formatted_query)

    results = get_collection().query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where={"chat_id": chat_id},
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    return [
        {
            "id": ids[index] if index < len(ids) else None,
            "content": documents[index] if index < len(documents) else "",
            "metadata": metadatas[index] if index < len(metadatas) else {},
            "distance": distances[index] if index < len(distances) else None,
        }
        for index in range(len(documents))
    ]


def delete_document_vectors(document_id: int) -> None:
    """Delete every vector associated with a specific document."""
    get_collection().delete(where={"document_id": document_id})


def delete_chat_data(chat_id: int) -> None:
    """Delete every vector associated with a chat."""
    get_collection().delete(where={"chat_id": chat_id})
