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

# Lazy-loaded Gemini embeddings with task-type support
_document_embeddings: GoogleGenerativeAIEmbeddings | None = None
_query_embeddings: GoogleGenerativeAIEmbeddings | None = None


def _get_document_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the document-task embeddings (index-time)."""
    global _document_embeddings
    if _document_embeddings is None:
        api_key = SecretStr(settings.gemini_api_key) if settings.gemini_api_key else None
        _document_embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=api_key,
            task_type="retrieval_document",
        )
    return _document_embeddings


def _get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the query-task embeddings (query-time)."""
    global _query_embeddings
    if _query_embeddings is None:
        api_key = SecretStr(settings.gemini_api_key) if settings.gemini_api_key else None
        _query_embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=api_key,
            task_type="retrieval_query",
        )
    return _query_embeddings


class _GeminiEmbeddingFunction:
    """Wrap Gemini embeddings for Chroma's embedding_function interface."""

    def name(self) -> str:
        return "gemini-embedding-001"

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed documents using the document task type."""
        return _get_document_embeddings().embed_documents(input)


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


def add_document_chunks(chunks: list[str], chat_id: int, document_id: int) -> list[str]:
    """Insert document chunks into the global collection with per-chunk metadata."""
    if not chunks:
        return []

    ids = [f"doc_{document_id}_chunk_{index}" for index, _ in enumerate(chunks, start=1)]
    metadatas = [
        {"chat_id": chat_id, "document_id": document_id}
        for _ in chunks
    ]

    get_collection().add(documents=chunks, metadatas=metadatas, ids=ids)
    return ids


def query_chat_documents(query_text: str, chat_id: int, n_results: int = 5) -> list[dict[str, Any]]:
    """Query the global collection, restricted to the active chat via metadata filter."""
    if not query_text.strip():
        return []

    # Embed the query manually using the retrieval_query task type, then pass
    # query_embeddings directly so ChromaDB does not try to call embed_query
    # on the collection's embedding function (which only handles documents).
    query_vector = _get_query_embeddings().embed_query(query_text)

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


def delete_chat_data(chat_id: int) -> None:
    """Delete every vector associated with a chat."""
    get_collection().delete(where={"chat_id": chat_id})
