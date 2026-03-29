# MCP Book Library Server

An MCP server for querying your personal ebook library using semantic search. Drop books into a folder, and the system converts, chunks, embeds (Gemini), auto-categorizes, and indexes them for natural language queries.

## Features

- **ingest_books** — Scan a folder for ebooks, convert to text, chunk, embed, auto-categorize
- **search_books** — Semantic search across entire library ("what does the author say about risk?")
- **ask_about_book** — Query a specific book
- **list_books** — Browse library by category
- **get_book_summary** — Read the opening of any book
- **get_library_status** — Library statistics

## Supported Formats

EPUB, MOBI, AZW3, AZW, PDF, TXT — requires [Calibre](https://calibre-ebook.com/) for conversion.

## How It Works

1. Drop an ebook file into the inbox folder (`/Volumes/Documents/BookInbox/`)
2. Call `ingest_books` — the system:
   - Converts to plain text via Calibre's `ebook-convert`
   - Splits into ~1500-character chunks with overlap
   - Generates embeddings via Gemini (`gemini-embedding-exp-03-07`)
   - Auto-categorizes via Gemini (`gemini-2.0-flash`)
   - Stores in ChromaDB (vectors) + SQLite (metadata)
   - Moves the original to `processed/`
3. Search with natural language queries

## Prerequisites

- Python 3.11+
- [Calibre](https://calibre-ebook.com/) installed (provides `ebook-convert`)
- A Gemini API key (for embeddings and classification)

## Installation

```bash
git clone https://github.com/locupleto/mcp-book-library.git
cd mcp-book-library
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Add to your `.mcp.json`:

```json
{
  "book-library": {
    "type": "stdio",
    "command": "/path/to/mcp-book-library/venv/bin/python3",
    "args": ["/path/to/mcp-book-library/book_library_mcp_server.py"],
    "env": {
      "GEMINI_API_KEY": "your-gemini-api-key",
      "BOOK_LIBRARY_INBOX": "/Volumes/Documents/BookInbox"
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Gemini API key |
| `BOOK_LIBRARY_DB_PATH` | `~/.book-library/book_library.db` | SQLite database |
| `BOOK_LIBRARY_CHROMA_PATH` | `~/.book-library/chroma` | ChromaDB storage |
| `BOOK_LIBRARY_INBOX` | `/Volumes/Documents/BookInbox` | Drop books here |
| `BOOK_LIBRARY_PROCESSED` | `<inbox>/processed` | Processed originals |
| `CALIBRE_EBOOK_CONVERT` | `/opt/homebrew/bin/ebook-convert` | Calibre converter |

## Auto-Categorization

Books are automatically classified into one of: Finance, Trading, Philosophy, Technology, Health, Fiction, Self-Help, Science, History, Business, Programming, Psychology, Spirituality, Education, Cooking, Other.

No manual tagging needed — Gemini reads the first few pages and assigns a category.

## Testing

```bash
source venv/bin/activate
python test_server.py
```

## DRM-Protected Books

If `ebook-convert` fails on a DRM-protected book, process it through the Calibre GUI first (with the DeDRM plugin installed), then drop the converted file into the inbox.
