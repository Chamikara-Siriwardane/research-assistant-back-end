# Backend Functionality Overview

This backend is an agentic research assistant that manages multi-turn chats, stores uploaded PDFs, indexes document pages for retrieval, and streams researched answers back to the client in real time. It is not just a CRUD API. The important behavior is the end-to-end workflow:

1. A user creates or opens a chat session.
2. The user uploads PDFs into that chat.
3. The system ingests the PDFs into a vector store after splitting them into pages.
4. The user sends a question.
5. A multi-agent pipeline decides whether to answer from uploaded documents, mocked web research, or mocked code analysis.
6. The final response is streamed token by token and saved back to the chat history.

## What the backend is designed to do

The application is a research assistant built around retrieval-augmented generation and streaming responses. Its core job is to keep a chat session organized, attach uploaded knowledge to that session, and use that knowledge during future answers.

The backend keeps three kinds of state:

- Chat metadata, such as title and timestamps.
- Message history, storing both user messages and assistant replies.
- Uploaded documents, their ingestion status, and the S3 location where the original file is kept.

It also maintains a persistent Chroma collection for semantic retrieval. Document vectors are stored with chat-scoped metadata so one chat does not leak knowledge into another.

## User-facing behavior

### Chat sessions

A chat is the container for a conversation. The backend lets the client create a new chat, list existing chats, and fetch a chat in full.

The chat list is ordered by the most recently updated session, so active conversations surface first. When a new chat is created, it starts with a default title of New Chat and no messages or documents.

The first message in a newly created chat triggers automatic title generation. The backend asks a lightweight Gemini model to summarize the first user message into a short title, then updates the chat record and emits a title update event to the client.

### Streaming answers

The main interaction is a streaming message endpoint. When the user sends a question, the backend:

- Stores the user message in SQLite.
- Loads a sliding window of recent messages from that chat.
- Converts them into LangChain message objects.
- Runs the multi-agent research pipeline.
- Streams intermediate reasoning updates and answer tokens over Server-Sent Events.
- Saves the final assistant reply back to SQLite.

This means the client does not wait for a single blocking response. Instead, it can show the assistant thinking, show answer tokens as they arrive, and preserve the full conversation after completion.

### Document uploads

Users can attach PDF documents to a chat. A successful upload immediately creates a document record with status processing and returns before ingestion is finished.

The actual upload flow is intentionally chat-specific:

- Only PDF files are accepted.
- The file is uploaded to S3 under a chat-specific prefix.
- The document row stores the S3 URL, file name, and processing status.
- A background task performs ingestion after the request returns.

This design keeps document handling tied to the chat that owns it, which is important because retrieval is later filtered by chat ID.

### Document readiness and access

Uploaded documents can be polled for their processing status. The possible states are processing, ready, and failed.

Once a document is ready, the backend can generate a presigned S3 URL so the client can download or preview the source PDF without exposing long-lived storage credentials.

## How document ingestion works

Document ingestion is more than file storage. The backend turns each PDF into retrievable page-level knowledge.

The ingestion pipeline works like this:

1. Download the uploaded PDF from S3.
2. Split the PDF into single-page PDF byte streams.
3. Send each page to Gemini multimodal embedding.
4. Store the resulting vectors in ChromaDB.
5. Mark the document ready when ingestion completes.

The important detail is that the backend does not extract plain text from the PDF during ingestion. It keeps the original PDF structure and uses multimodal embeddings on the raw page bytes. That gives the retrieval layer page-level granularity while preserving layout and document structure.

Each stored vector is tagged with metadata such as:

- chat_id
- document_id
- page_number
- chunk_type

That metadata is what allows retrieval to stay scoped to the correct conversation.

If ingestion fails at any point, the document is marked failed so the client can distinguish a real processing problem from a document that is still being handled.

## How the agent pipeline works

The answer generation flow is a small research workflow built with LangGraph. The graph is not a single prompt. It is a routed pipeline with specialist roles.

### 1. Supervisor

The supervisor reads the latest user query and the recent conversation history, then decides which specialist should handle the request next.

It can route to:

- Librarian for uploaded documents.
- Scout for web-style research.
- Analyst for code or calculation tasks.
- Synthesizer when enough context already exists.

The supervisor also has retry awareness. If the critic rejects an earlier retrieval attempt, the supervisor is prompted to choose a different route unless the current evidence is already good enough.

### 2. Librarian

The librarian is the real retrieval node in the system. It searches the persistent Chroma collection using the active chat ID as a filter, so it only sees documents attached to the current conversation.

Its behavior is:

- Run semantic retrieval against the chat-scoped vector store.
- Log the ranked matching page snippets.
- Resolve the source PDFs from S3.
- Slice the matching PDF pages back out as raw PDF bytes.
- Add those pages to the agent state for multimodal synthesis.

This is the part of the backend that makes uploaded PDFs feel like searchable private knowledge instead of just attached files.

### 3. Critic

The critic checks whether the retrieved context is actually good enough to answer the question.

It is intentionally not overly strict. The goal is to avoid endless retry loops. If the context covers the core of the query, the critic can approve it and let synthesis proceed. If the context is too weak or off-topic, the critic can force another retrieval cycle.

This makes the pipeline more robust than a one-pass retrieval flow because it can re-route when retrieval quality is poor.

### 4. Scout

The scout currently behaves as a placeholder for live web research. It returns mock web-style snippets after a short delay. The architecture clearly reserves this role for future real web search integration, but in the current code it does not call an external search provider.

### 5. Analyst

The analyst currently behaves as a placeholder for code execution or computational analysis. It returns mock execution-style output after a short delay. Like the scout, it represents a planned capability more than a real external integration in the present codebase.

### 6. Synthesizer

The synthesizer writes the final answer.

It builds a prompt from:

- Recent conversation history.
- The latest user question.
- Retrieved context from the specialist nodes.
- Raw PDF pages, when the librarian found document pages worth passing through.

When PDF pages are available, the synthesizer uses a multimodal Gemini prompt and passes the raw page bytes directly. When no pages are available, it falls back to a text-only prompt using the accumulated context.

The synthesizer streams the answer token by token so the client can render output progressively. It is also instructed to produce structured Markdown, answer directly first, and cite source labels inline when possible.

## What gets stored in the database

The SQLite database is the system of record for chat state.

### Chats

Chat rows store the session title and timestamps. The title is updated automatically when the first user message arrives if the chat still has a default title.

### Messages

Messages are stored with:

- chat_id
- sender_type
- content
- timestamp

The system stores both user and assistant messages, which makes the conversation resumable and enables context-aware prompting on later turns.

### Documents

Documents are stored with:

- chat_id
- file_name
- s3_url
- status
- uploaded_at

The document record is the bridge between the chat session, the S3 object, and the retrieval index.

## How retrieval stays chat-specific

The vector store is persistent, but it is not global in a way that mixes conversations together.

All indexed content is stored in one Chroma collection named jarvis_global_knowledge, but every record carries chat_id and document_id metadata. Retrieval queries filter on chat_id, so only documents uploaded to the active chat are considered.

That means the system behaves like a per-chat private knowledge base even though the physical storage is shared.

## Streaming event behavior

The streaming endpoint does not only send final answer text. It emits several event types that describe what the assistant is doing.

- title_update when a new chat title is generated.
- thought when a specialist node starts work.
- text for streamed answer tokens.
- error when the pipeline reports a fatal problem.
- [DONE] at the end of the stream.

This is what makes the UI feel active and transparent. The client can display routing, retrieval, and response generation as they happen rather than waiting for a silent backend round-trip.

## Configuration and runtime assumptions

The backend is configured through environment variables or a .env file.

Important runtime dependencies include:

- Gemini API key and model settings for routing, embedding, and answer generation.
- AWS credentials and region for document storage in S3.
- A writable local SQLite database file.
- A writable local chroma_data directory for persistent vector storage.

The backend also enables CORS for the configured frontend origin so a browser-based client can talk to it during development.

## Startup and operational behavior

When the application starts, it configures logging, attaches console handlers if needed, and creates the SQLite tables from the SQLAlchemy models. That means the backend is designed to self-initialize its database schema on startup instead of relying on a separate migration step in the current prototype.

It also exposes a lightweight health check that reports the service status and version. This is the simplest way to confirm the process is alive and the configured version is running.

The startup path is intentionally simple:

- load settings
- configure logging
- create database tables if they do not exist
- mount the API routers
- serve the app with CORS enabled for the frontend origin

## Current implementation boundaries

Some capabilities are fully implemented and some are only scaffolded.

Implemented now:

- Chat creation, listing, and full chat retrieval.
- Message streaming with persistent storage.
- Automatic chat title generation.
- PDF upload to S3.
- Background PDF ingestion and page-level multimodal embeddings.
- Chat-scoped vector retrieval from ChromaDB.
- Multimodal synthesis over retrieved PDF pages.
- Document status tracking and presigned URL generation.

Scaffolded or mocked now:

- Web search in the Scout node.
- Code execution in the Analyst node.

Those two nodes already fit into the architecture, but their current bodies return mock context rather than calling real external tools.

## What this backend is good at

The completed functionality is best understood as a private research assistant for chat-specific document collections. It works well when the user wants to upload source PDFs, ask follow-up questions about them, and receive streamed, context-aware answers that cite the retrieved pages.

It is not just a document upload service. It is a full conversation system with persistent memory, ingestion, retrieval, routing, quality control, and streamed synthesis.

## Practical end-to-end example

In practice, a user flow looks like this:

1. Create a new chat.
2. Upload one or more PDFs to that chat.
3. Wait until the documents report ready.
4. Ask a question about the uploaded material.
5. The supervisor selects the librarian because the question appears document-focused.
6. The librarian retrieves relevant pages from the vector store and S3.
7. The critic checks whether the retrieved pages are sufficient.
8. The synthesizer writes a final Markdown answer and streams it back to the client.
9. The answer is saved to the chat history.

That is the real functional shape of the backend.