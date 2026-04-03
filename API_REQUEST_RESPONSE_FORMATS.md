# API Request and Response Formats

This document covers the following endpoints:
- `/api/chats/`
- `/api/chats/{chat_id}/documents`
- `/api/documents/{document_id}/status`

## 1) `/api/chats/`

### GET `/api/chats/`
Returns all chats ordered by most recently updated.

Request:
- Method: `GET`
- Body: none
- Path params: none
- Query params: none

Success Response:
- Status: `200 OK`
- Content-Type: `application/json`
- Body:
```json
[
  {
    "id": 3,
    "title": "New Chat",
    "updated_at": "2026-04-03T10:42:51.123456"
  }
]
```

Schema:
- `id`: integer
- `title`: string
- `updated_at`: datetime string

### POST `/api/chats/`
Creates a new chat with default title (no request payload required).

Request:
- Method: `POST`
- Body: none
- Path params: none
- Query params: none

Success Response:
- Status: `201 Created`
- Content-Type: `application/json`
- Body:
```json
{
  "chat_id": 4,
  "title": "New Chat",
  "created_at": "2026-04-03T10:45:12.654321"
}
```

Schema:
- `chat_id`: integer
- `title`: string
- `created_at`: datetime string

## 2) `/api/chats/{chat_id}/documents`

### POST `/api/chats/{chat_id}/documents`
Uploads one PDF file to S3, creates a `documents` row with `processing`, and starts async ingestion.

Request:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Path params:
  - `chat_id` (integer)
- Form fields:
  - `file` (required, file upload)
- Validation:
  - only `.pdf` files are accepted

Example (curl):
```bash
curl -X POST "http://localhost:8000/api/chats/3/documents" \
  -F "file=@Invoice.pdf"
```

Success Response:
- Status: `202 Accepted`
- Content-Type: `application/json`
- Body:
```json
{
  "document_id": 3,
  "file_name": "Invoice.pdf",
  "status": "processing"
}
```

Schema:
- `document_id`: integer
- `file_name`: string
- `status`: string (`processing` at this stage)

Error Responses:
- `400 Bad Request`
```json
{ "detail": "Only PDF files are supported." }
```
- `404 Not Found`
```json
{ "detail": "Chat not found" }
```
- `500 Internal Server Error`
```json
{ "detail": "Failed to upload document to S3: <provider error>" }
```

## 3) `/api/documents/{document_id}/status`

### GET `/api/documents/{document_id}/status`
Returns current ingestion status of a document.

Request:
- Method: `GET`
- Body: none
- Path params:
  - `document_id` (integer)
- Query params: none

Success Response:
- Status: `200 OK`
- Content-Type: `application/json`
- Body:
```json
{
  "document_id": 3,
  "status": "ready"
}
```

Possible `status` values in current pipeline:
- `processing`
- `ready`
- `failed`

Error Response:
- `404 Not Found`
```json
{ "detail": "Document not found" }
```
