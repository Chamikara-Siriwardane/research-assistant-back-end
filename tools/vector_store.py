"""
tools/vector_store.py
---------------------
Local vector database layer built on ChromaDB + LangChain.

Responsibilities
-----------------
1.  Hold a single, lazily-initialised Chroma collection so that every agent
    call shares the same in-process client (no extra HTTP server required).
2.  Expose helpers for seeding and appending documents to the collection.
3.  Expose `get_retriever()` so LangGraph nodes can drop it in as a tool
    without knowing anything about ChromaDB internals.

Persistence
-----------
Documents are written to ./chroma_db_data on disk.  Delete that directory to
start with a fresh collection.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from core.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "research_documents"
PERSIST_DIRECTORY = "./chroma_db_data"

# ---------------------------------------------------------------------------
# Embeddings — reuse a single instance across the process lifetime
# ---------------------------------------------------------------------------

_google_api_key = SecretStr(settings.gemini_api_key) if settings.gemini_api_key else None

_document_embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.embedding_model,
    google_api_key=_google_api_key,
    task_type="retrieval_document",
)

_query_embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.embedding_model,
    google_api_key=_google_api_key,
    task_type="retrieval_query",
)


class _GeminiRetrievalEmbeddings(Embeddings):
    """Use dedicated Gemini task types for index-time and query-time embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _document_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return _query_embeddings.embed_query(text)


_embeddings: Embeddings = _GeminiRetrievalEmbeddings()

# ---------------------------------------------------------------------------
# Chroma vector store — lazy singleton
# ---------------------------------------------------------------------------

_vector_store: Chroma | None = None


def get_vector_store() -> Chroma:
    """
    Return the shared Chroma instance, creating it on the first call.

    Using a module-level singleton avoids re-opening the SQLite file on
    every agent invocation, which would be both slow and unsafe under
    concurrent requests.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )
    return _vector_store


def add_documents(documents: list[Document]) -> None:
    """Append a batch of LangChain documents to the shared collection."""
    if not documents:
        return

    get_vector_store().add_documents(documents)


# ---------------------------------------------------------------------------
# Mock ingestion
# ---------------------------------------------------------------------------

#: Dummy documents that simulate a small private research corpus.
#: Each one carries structured metadata so we can later demonstrate
#: metadata-filtered retrieval (e.g. "only papers from 2024 in AI").
_MOCK_DOCUMENTS: list[Document] = [
    Document(
        page_content=(
            "Large Language Models (LLMs) have demonstrated remarkable in-context "
            "learning capabilities, allowing them to perform new tasks from just a "
            "few examples without any gradient updates.  This survey examines "
            "theoretical explanations for this behaviour across GPT-4, Claude 3, "
            "and Gemini Ultra, and outlines open research challenges."
        ),
        metadata={
            "source": "arxiv:2405.00001",
            "title": "A Survey of In-Context Learning in Large Language Models",
            "year": 2024,
            "category": "ai_research",
            "authors": "Zhang et al.",
        },
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) addresses the knowledge cutoff "
            "limitation of static LLMs by dynamically fetching relevant passages "
            "from an external corpus at inference time.  Recent work on Agentic RAG "
            "introduces autonomous tool-calling agents that iteratively refine "
            "queries, improving answer faithfulness by 18 % on RAGAS benchmarks."
        ),
        metadata={
            "source": "arxiv:2405.00002",
            "title": "Agentic RAG: Iterative Retrieval with Autonomous Agents",
            "year": 2024,
            "category": "ai_research",
            "authors": "Patel & Kim",
        },
    ),
    Document(
        page_content=(
            "Apple Inc. reported Q2 FY2024 revenue of $90.8 billion, a 4 % "
            "year-over-year increase driven by Services growth of 14 %.  iPhone "
            "revenue was $46.0 billion.  The company returned $27 billion to "
            "shareholders through dividends and buybacks and guided Q3 FY2024 "
            "revenue in the $84–87 billion range."
        ),
        metadata={
            "source": "apple_10q_q2_2024",
            "title": "Apple Q2 FY2024 Earnings Report",
            "year": 2024,
            "category": "financial_report",
            "authors": "Apple Inc. Investor Relations",
        },
    ),
    Document(
        page_content=(
            "NVIDIA Corporation posted record quarterly revenue of $26.0 billion "
            "for Q1 FY2025, up 262 % year-over-year, with Data Center revenue "
            "reaching $22.6 billion.  The surge was attributed to explosive demand "
            "for Hopper GPU infrastructure across hyperscalers and sovereign AI "
            "programmes.  Gross margin expanded to 78.4 %."
        ),
        metadata={
            "source": "nvidia_10q_q1_2025",
            "title": "NVIDIA Q1 FY2025 Earnings Report",
            "year": 2024,
            "category": "financial_report",
            "authors": "NVIDIA Investor Relations",
        },
    ),
    Document(
        page_content=(
            "Graph Neural Networks (GNNs) have become the dominant approach for "
            "molecular property prediction in drug discovery.  This paper proposes "
            "PharmaGNN, a heterogeneous GNN architecture that encodes atom, bond, "
            "and pharmacophore features jointly, achieving state-of-the-art results "
            "on the MoleculeNet benchmark suite with 12 % fewer parameters than "
            "prior work."
        ),
        metadata={
            "source": "arxiv:2405.00003",
            "title": "PharmaGNN: Heterogeneous Graph Networks for Drug Discovery",
            "year": 2025,
            "category": "ai_research",
            "authors": "Gupta et al.",
        },
    ),
]


def ingest_mock_data() -> None:
    """
    Add the mock research documents to the Chroma collection.

    Idempotency note: Chroma's `add_documents` assigns a UUID to each document
    by default, so calling this function multiple times will create duplicate
    embeddings.  For development, delete ./chroma_db_data between runs or guard
    with a collection-count check (see comment below).
    """
    store = get_vector_store()

    # Optional idempotency guard — skip ingestion if documents already exist.
    existing_count = store._collection.count()  # noqa: SLF001
    if existing_count > 0:
        print(f"[vector_store] Collection already contains {existing_count} documents — skipping ingestion.")
        return

    store.add_documents(_MOCK_DOCUMENTS)
    print(f"[vector_store] Ingested {len(_MOCK_DOCUMENTS)} mock documents into '{COLLECTION_NAME}'.")


# ---------------------------------------------------------------------------
# Retriever factory — the interface consumed by LangGraph nodes
# ---------------------------------------------------------------------------


def get_retriever(k: int = 2):
    """
    Return a LangChain BaseRetriever backed by the Chroma collection.

    Parameters
    ----------
    k:
        Number of top similar chunks to return per query.  Defaults to 2 so
        that the Critic has focused, high-relevance context to evaluate.

    Usage in an agent node
    ----------------------
    >>> retriever = get_retriever()
    >>> docs = await retriever.ainvoke("transformer attention mechanism")
    """
    return get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ---------------------------------------------------------------------------
# Standalone test — run with:  python tools/vector_store.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        print("=== ChromaDB Vector Store — Isolation Test ===\n")

        # 1. Seed the collection
        ingest_mock_data()

        # 2. Run a test retrieval
        test_query = "retrieval augmented generation agent"
        print(f"\nQuery: '{test_query}'")
        print("-" * 50)

        retriever = get_retriever(k=2)
        results: list[Document] = await retriever.ainvoke(test_query)

        for i, doc in enumerate(results, start=1):
            print(f"\n[Result {i}]")
            print(f"  Title   : {doc.metadata.get('title', 'N/A')}")
            print(f"  Source  : {doc.metadata.get('source', 'N/A')}")
            print(f"  Year    : {doc.metadata.get('year', 'N/A')}")
            print(f"  Category: {doc.metadata.get('category', 'N/A')}")
            print(f"  Snippet : {doc.page_content[:120]}...")

        print("\n=== Test complete ===")

    asyncio.run(_main())
