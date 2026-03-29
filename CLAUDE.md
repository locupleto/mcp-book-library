# CLAUDE.md

## Overview

MCP server for querying a personal ebook library. Converts books to text via Calibre, chunks and embeds with Gemini, auto-categorizes, stores in ChromaDB for semantic search.

## Tools

| Tool | Use When |
|------|----------|
| `ingest_books` | Adding new books — scan inbox or custom folder |
| `search_books` | Finding passages across all books by meaning |
| `ask_about_book` | Querying a specific book |
| `list_books` | Browsing the library |
| `get_book_summary` | Reading a book's opening |
| `get_library_status` | Checking library health and stats |

## Architecture

- **SQLite** (`~/.book-library/book_library.db`) — book metadata + full text
- **ChromaDB** (`~/.book-library/chroma/`) — chunked text + Gemini embeddings
- **Gemini** — `gemini-embedding-exp-03-07` for embeddings, `gemini-2.0-flash` for classification
- **Calibre** — `ebook-convert` for format conversion (EPUB, MOBI, AZW3, PDF → TXT)

## Workflow

1. User drops ebook in `/Volumes/Documents/BookInbox/`
2. `ingest_books()` converts, chunks, embeds, categorizes, indexes
3. `search_books(query)` finds semantically relevant passages

## Environment

- `GEMINI_API_KEY` — Gemini API key (from env)
- `BOOK_LIBRARY_DB_PATH` — SQLite path
- `BOOK_LIBRARY_CHROMA_PATH` — ChromaDB path
- `BOOK_LIBRARY_INBOX` — where to drop new books
- `CALIBRE_EBOOK_CONVERT` — path to Calibre's ebook-convert
