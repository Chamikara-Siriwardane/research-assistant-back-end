"""
api/cache.py
------------
Lightweight in-process caches shared between API modules.

Keeps per-chat state that is expensive to re-derive on every request but
safe to hold in memory for the lifetime of the server process.
"""

# chat_id → bool: True  if the chat has at least one document with status='ready'
#                 False if the chat has been checked and has no ready documents
#                 absent if the result has never been queried (cache miss)
_has_documents: dict[int, bool] = {}


def get_has_documents(chat_id: int) -> bool | None:
    """Return the cached value, or None on a cache miss."""
    return _has_documents.get(chat_id)


def set_has_documents(chat_id: int, value: bool) -> None:
    """Store the result of a DB lookup so subsequent requests skip the query."""
    _has_documents[chat_id] = value


def invalidate_has_documents(chat_id: int) -> None:
    """
    Drop the cached entry for *chat_id*.

    Call this whenever a document for the chat transitions to 'ready' so the
    next request re-queries the DB and picks up the newly available document.
    """
    _has_documents.pop(chat_id, None)
