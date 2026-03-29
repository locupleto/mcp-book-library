#!/usr/bin/env python3
"""
MCP Server for Personal Book Library

Ingest ebooks (EPUB, MOBI, AZW3, PDF, TXT), convert to text via Calibre,
chunk and embed with Gemini, auto-categorize, and provide semantic search
across your entire library.

Tools:
- ingest_books: Scan inbox folder for new books, convert, chunk, embed, categorize
- search_books: Semantic search across all books
- ask_about_book: Find relevant passages from a specific book
- list_books: List all books with metadata
- get_book_summary: First chunks of a book (intro/preface)
- get_library_status: Library statistics

Environment Variables:
- GEMINI_API_KEY: Google Gemini API key for embeddings and classification
- BOOK_LIBRARY_DB_PATH: Path to SQLite database
- BOOK_LIBRARY_CHROMA_PATH: Path to ChromaDB persistence directory
- BOOK_LIBRARY_INBOX: Path to inbox folder for new books
- BOOK_LIBRARY_PROCESSED: Path to move processed originals
- CALIBRE_EBOOK_CONVERT: Path to Calibre's ebook-convert binary
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("book-library-mcp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DB_PATH = os.environ.get("BOOK_LIBRARY_DB_PATH",
                         os.path.expanduser("~/.book-library/book_library.db"))
CHROMA_PATH = os.environ.get("BOOK_LIBRARY_CHROMA_PATH",
                             os.path.expanduser("~/.book-library/chroma"))
INBOX_PATH = os.environ.get("BOOK_LIBRARY_INBOX",
                            "/Volumes/Documents/BookLibrary/inbox")
COLLECTIONS_PATH = os.environ.get("BOOK_LIBRARY_COLLECTIONS",
                                  "/Volumes/Documents/BookLibrary/collections")
EBOOK_CONVERT = os.environ.get("CALIBRE_EBOOK_CONVERT",
                               "/opt/homebrew/bin/ebook-convert")

SUPPORTED_FORMATS = {'.epub', '.mobi', '.azw3', '.azw', '.pdf', '.txt'}
EMBEDDING_MODEL = "gemini-embedding-001"
CLASSIFICATION_MODEL = "gemini-2.0-flash"
CHUNK_SIZE = 1500  # characters
CHUNK_OVERLAP = 200  # characters
EMBEDDING_BATCH_SIZE = 100

VALID_CATEGORIES = [
    "Finance", "Trading", "Philosophy", "Technology", "Health", "Fiction",
    "Self-Help", "Science", "History", "Business", "Programming",
    "Psychology", "Spirituality", "Education", "Cooking", "Other"
]

# Initialize MCP server
server = Server("book-library")

# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

_genai_client = None


def get_genai_client():
    """Get or create the Gemini client."""
    global _genai_client
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


# ---------------------------------------------------------------------------
# SQLite Database Layer
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize database with schema."""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT DEFAULT 'Unknown',
            file_path TEXT NOT NULL,
            original_format TEXT NOT NULL,
            category TEXT DEFAULT 'Uncategorized',
            date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
            chunk_count INTEGER DEFAULT 0,
            file_hash TEXT UNIQUE NOT NULL,
            file_size_bytes INTEGER,
            ingestion_status TEXT DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS book_texts (
            book_id INTEGER PRIMARY KEY REFERENCES books(id),
            full_text TEXT NOT NULL,
            text_length INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_books_category ON books(category);
        CREATE INDEX IF NOT EXISTS idx_books_file_hash ON books(file_hash);
        CREATE INDEX IF NOT EXISTS idx_books_title ON books(title);
    """)
    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    """Get a database connection."""
    return init_db(DB_PATH)


# ---------------------------------------------------------------------------
# ChromaDB Layer
# ---------------------------------------------------------------------------

_chroma_client = None
_chroma_collection = None


def get_chroma_collection():
    """Get or create the ChromaDB collection."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        import chromadb
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name="book_chunks",
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection


# ---------------------------------------------------------------------------
# Gemini Embeddings
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Gemini API in batches."""
    client = get_genai_client()
    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch
        )
        all_embeddings.extend([e.values for e in result.embeddings])
        if i + EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(0.5)  # Rate limit protection

    return all_embeddings


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for a single search query."""
    client = get_genai_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query]
    )
    return result.embeddings[0].values


# ---------------------------------------------------------------------------
# Auto-Categorization
# ---------------------------------------------------------------------------

def classify_book(text_sample: str, title: str, author: str) -> str:
    """Use Gemini to classify a book into a category."""
    from google.genai import types

    client = get_genai_client()
    categories_str = ", ".join(VALID_CATEGORIES)
    prompt = f"""Classify this book into exactly ONE category from this list:
{categories_str}

Title: {title}
Author: {author}
Text sample (first ~3000 characters):
{text_sample[:3000]}

Respond with ONLY the category name, nothing else."""

    response = client.models.generate_content(
        model=CLASSIFICATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )
    category = response.text.strip()
    return category if category in VALID_CATEGORIES else "Other"


# ---------------------------------------------------------------------------
# Text Processing
# ---------------------------------------------------------------------------

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_metadata_from_filename(file_path: str) -> tuple[str, str]:
    """Extract title and author from filename. Returns (title, author)."""
    name = Path(file_path).stem

    # Strip common artifacts
    artifacts = [
        r'\(\s*PDFDrive\s*\)', r'\(\s*PDFDrive\.com\s*\)',
        r'\(\s*book-drive\.com\s*\)', r'\[\s*Studycrux\.com\s*\]',
        r'\(\s*z-lib\.org\s*\)', r'\(\s*www\.ebook-dl\.com\s*\)',
        r'\(\s*\d+\s*\)',  # (2023) year in parens
    ]
    for pattern in artifacts:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()

    # Try to split author/title on common separators
    for sep in [' - ', ' — ', ' – ', ' by ']:
        if sep in name:
            parts = name.split(sep, 1)
            # Convention: "Title - Author" or "Author - Title"
            # Heuristic: shorter part is more likely the author
            if len(parts[0]) < len(parts[1]):
                return parts[1].strip(), parts[0].strip()
            else:
                return parts[0].strip(), parts[1].strip()

    return name, "Unknown"


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                           overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
            if chunk.strip():
                chunks.append(chunk.strip())
            break

        # Try to find a natural break point
        best_break = end
        for sep in separators:
            # Search backwards from end for the separator
            pos = text.rfind(sep, start + chunk_size // 2, end)
            if pos != -1:
                best_break = pos + len(sep)
                break

        chunk = text[start:best_break]
        if chunk.strip():
            chunks.append(chunk.strip())

        # Move start forward, accounting for overlap
        start = best_break - overlap
        if start <= chunks[-1] if not chunks else 0:
            start = best_break  # Prevent infinite loop

    return chunks


# ---------------------------------------------------------------------------
# Book Conversion
# ---------------------------------------------------------------------------

def convert_to_text(file_path: str) -> str:
    """Convert an ebook to plain text using Calibre's ebook-convert."""
    ext = Path(file_path).suffix.lower()

    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    # Use ebook-convert for all other formats
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [EBOOK_CONVERT, file_path, tmp_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"ebook-convert failed: {result.stderr[:500]}")

        with open(tmp_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()

        if len(text.strip()) < 100:
            raise RuntimeError("Conversion produced very little text — file may be DRM-protected or scanned PDF")

        return text
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Ingestion Pipeline
# ---------------------------------------------------------------------------

def ingest_single_book(file_path: str, conn: sqlite3.Connection) -> dict:
    """Ingest a single book file. Returns result dict with status and details."""
    file_path = os.path.abspath(file_path)
    ext = Path(file_path).suffix.lower()
    file_size = os.path.getsize(file_path)

    # Calculate hash for dedup
    file_hash = calculate_file_hash(file_path)

    # Check for duplicate
    existing = conn.execute("SELECT title FROM books WHERE file_hash=?", (file_hash,)).fetchone()
    if existing:
        return {"status": "skipped", "reason": f"Duplicate of '{existing[0]}'", "file": Path(file_path).name}

    # Extract metadata
    title, author = extract_metadata_from_filename(file_path)

    # Insert book record (pending)
    cursor = conn.execute(
        """INSERT INTO books (title, author, file_path, original_format, file_hash,
           file_size_bytes, ingestion_status) VALUES (?, ?, ?, ?, ?, ?, 'processing')""",
        (title, author, file_path, ext.lstrip('.'), file_hash, file_size)
    )
    book_id = cursor.lastrowid
    conn.commit()

    try:
        # Convert to text
        logger.info(f"Converting {Path(file_path).name} to text...")
        text = convert_to_text(file_path)

        # Store full text
        conn.execute(
            "INSERT INTO book_texts (book_id, full_text, text_length) VALUES (?, ?, ?)",
            (book_id, text, len(text))
        )

        # Chunk
        chunks = split_text_into_chunks(text)
        if not chunks:
            raise RuntimeError("No text chunks produced")

        # Classify
        logger.info(f"Classifying '{title}'...")
        category = classify_book(text, title, author)

        # Embed
        logger.info(f"Embedding {len(chunks)} chunks for '{title}'...")
        embeddings = get_embeddings(chunks)

        # Store in ChromaDB
        collection = get_chroma_collection()
        ids = [f"{book_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "book_id": book_id,
                "book_title": title,
                "author": author,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

        # Update book record
        conn.execute(
            """UPDATE books SET category=?, chunk_count=?, ingestion_status='completed'
               WHERE id=?""",
            (category, len(chunks), book_id)
        )
        conn.commit()

        # Move to collections/<category>/
        category_dir = os.path.join(COLLECTIONS_PATH, category)
        os.makedirs(category_dir, exist_ok=True)
        dest = os.path.join(category_dir, Path(file_path).name)
        if os.path.abspath(file_path) != os.path.abspath(dest):
            counter = 1
            while os.path.exists(dest):
                stem = Path(file_path).stem
                dest = os.path.join(category_dir, f"{stem}_{counter}{ext}")
                counter += 1
            shutil.move(file_path, dest)

        logger.info(f"Ingested '{title}' [{category}] — {len(chunks)} chunks")
        return {
            "status": "completed", "title": title, "author": author,
            "category": category, "chunks": len(chunks), "file": Path(file_path).name
        }

    except Exception as e:
        conn.execute(
            "UPDATE books SET ingestion_status='failed' WHERE id=?", (book_id,)
        )
        conn.commit()
        error_msg = str(e)
        logger.error(f"Failed to ingest {Path(file_path).name}: {error_msg}")
        return {"status": "failed", "file": Path(file_path).name, "error": error_msg}


# ---------------------------------------------------------------------------
# MCP Tool Handlers
# ---------------------------------------------------------------------------

async def handle_ingest_books(arguments: dict) -> Sequence[TextContent]:
    """Scan inbox folder for new books and ingest them."""
    folder = arguments.get("folder_path", INBOX_PATH)
    recursive = arguments.get("recursive", False)

    if not os.path.isdir(folder):
        return [TextContent(type="text", text=f"Error: Folder not found: {folder}")]

    # Find supported files
    files = []
    if recursive:
        for root, _, filenames in os.walk(folder):
            # Skip processed folder
            if "processed" in root.lower():
                continue
            for f in filenames:
                if Path(f).suffix.lower() in SUPPORTED_FORMATS:
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if Path(f).suffix.lower() in SUPPORTED_FORMATS and f != "processed":
                files.append(os.path.join(folder, f))

    if not files:
        return [TextContent(type="text", text=f"No supported files found in {folder}.\nSupported formats: {', '.join(SUPPORTED_FORMATS)}")]

    conn = get_db()
    results = []
    for file_path in sorted(files):
        result = ingest_single_book(file_path, conn)
        results.append(result)

    conn.close()

    # Format summary
    completed = [r for r in results if r["status"] == "completed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]

    lines = [f"Ingestion complete: {len(completed)} added, {len(skipped)} skipped, {len(failed)} failed\n"]

    if completed:
        lines.append("Added:")
        for r in completed:
            lines.append(f"  + {r['title']} [{r['category']}] — {r['chunks']} chunks")

    if skipped:
        lines.append("\nSkipped (duplicates):")
        for r in skipped:
            lines.append(f"  ~ {r['file']} — {r['reason']}")

    if failed:
        lines.append("\nFailed:")
        for r in failed:
            lines.append(f"  ! {r['file']} — {r['error']}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_search_books(arguments: dict) -> Sequence[TextContent]:
    """Semantic search across the book library."""
    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="Error: 'query' parameter is required")]

    category = arguments.get("category")
    book_title = arguments.get("book_title")
    limit = min(arguments.get("limit", 5), 20)

    collection = get_chroma_collection()
    if collection.count() == 0:
        return [TextContent(type="text", text="Library is empty. Use ingest_books to add books first.")]

    # Build filter
    where = None
    if category and book_title:
        # Look up book_id from SQLite for title filter
        conn = get_db()
        rows = conn.execute(
            "SELECT id FROM books WHERE title LIKE ? AND ingestion_status='completed'",
            (f"%{book_title}%",)
        ).fetchall()
        conn.close()
        if rows:
            book_ids = [r[0] for r in rows]
            where = {"$and": [{"category": category}, {"book_id": {"$in": book_ids}}]}
        else:
            where = {"category": category}
    elif category:
        where = {"category": category}
    elif book_title:
        conn = get_db()
        rows = conn.execute(
            "SELECT id FROM books WHERE title LIKE ? AND ingestion_status='completed'",
            (f"%{book_title}%",)
        ).fetchall()
        conn.close()
        if rows:
            book_ids = [r[0] for r in rows]
            where = {"book_id": {"$in": book_ids}}

    # Embed query and search
    query_emb = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=limit,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        return [TextContent(type="text", text=f"No results found for '{query}'.")]

    lines = [f"Search results for '{query}' ({len(results['documents'][0])} matches):\n"]
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        score = 1 - dist  # cosine distance → similarity
        lines.append(f"--- Result {i+1} (relevance: {score:.2f}) ---")
        lines.append(f"Book: {meta['book_title']} by {meta['author']} [{meta['category']}]")
        lines.append(f"Position: chunk {meta['chunk_index']+1}/{meta['total_chunks']}")
        lines.append(f"\n{doc[:800]}{'...' if len(doc) > 800 else ''}\n")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_ask_about_book(arguments: dict) -> Sequence[TextContent]:
    """Find relevant passages from a specific book."""
    book_title = arguments.get("book_title", "").strip()
    question = arguments.get("question", "").strip()
    if not book_title:
        return [TextContent(type="text", text="Error: 'book_title' parameter is required")]
    if not question:
        return [TextContent(type="text", text="Error: 'question' parameter is required")]

    limit = min(arguments.get("limit", 5), 20)

    # Fuzzy match title in SQLite
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, author, category FROM books WHERE title LIKE ? AND ingestion_status='completed'",
        (f"%{book_title}%",)
    ).fetchall()
    conn.close()

    if not rows:
        return [TextContent(type="text", text=f"No book found matching '{book_title}'. Use list_books to see available titles.")]

    book_id, matched_title, author, category = rows[0]

    collection = get_chroma_collection()
    query_emb = get_query_embedding(question)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=limit,
        where={"book_id": book_id},
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        return [TextContent(type="text", text=f"No relevant passages found in '{matched_title}'.")]

    lines = [f"Passages from '{matched_title}' by {author} [{category}] relevant to: \"{question}\"\n"]
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        score = 1 - dist
        lines.append(f"--- Passage {i+1} (relevance: {score:.2f}, chunk {meta['chunk_index']+1}/{meta['total_chunks']}) ---")
        lines.append(f"{doc}\n")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_list_books(arguments: dict) -> Sequence[TextContent]:
    """List all books in the library."""
    category = arguments.get("category")

    conn = get_db()
    if category:
        rows = conn.execute(
            "SELECT title, author, category, date_added, chunk_count, ingestion_status FROM books WHERE category=? ORDER BY title",
            (category,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT title, author, category, date_added, chunk_count, ingestion_status FROM books ORDER BY category, title"
        ).fetchall()

    # Get categories summary
    categories = conn.execute(
        "SELECT category, COUNT(*) FROM books WHERE ingestion_status='completed' GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()

    if not rows:
        return [TextContent(type="text", text="Library is empty. Use ingest_books to add books.")]

    lines = []
    if categories:
        lines.append("Categories: " + ", ".join(f"{c} ({n})" for c, n in categories))
        lines.append("")

    current_cat = None
    for title, author, cat, date_added, chunks, status in rows:
        if cat != current_cat:
            current_cat = cat
            lines.append(f"\n**{cat}**")
        status_str = f" [{status}]" if status != "completed" else ""
        date_str = date_added[:10] if date_added else "?"
        lines.append(f"  - {title} — {author} ({chunks} chunks, {date_str}){status_str}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_book_summary(arguments: dict) -> Sequence[TextContent]:
    """Get the first few chunks of a book."""
    book_title = arguments.get("book_title", "").strip()
    if not book_title:
        return [TextContent(type="text", text="Error: 'book_title' parameter is required")]

    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, author, category FROM books WHERE title LIKE ? AND ingestion_status='completed'",
        (f"%{book_title}%",)
    ).fetchall()
    conn.close()

    if not rows:
        return [TextContent(type="text", text=f"No book found matching '{book_title}'.")]

    book_id, matched_title, author, category = rows[0]

    collection = get_chroma_collection()
    results = collection.get(
        where={"book_id": book_id},
        include=["documents", "metadatas"],
        limit=3
    )

    if not results["documents"]:
        return [TextContent(type="text", text=f"No content found for '{matched_title}'.")]

    # Sort by chunk_index
    paired = sorted(zip(results["documents"], results["metadatas"]), key=lambda x: x[1]["chunk_index"])

    lines = [f"Summary of '{matched_title}' by {author} [{category}]\n"]
    for doc, meta in paired:
        lines.append(f"--- Chunk {meta['chunk_index']+1}/{meta['total_chunks']} ---")
        lines.append(doc)
        lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_get_library_status(arguments: dict) -> Sequence[TextContent]:
    """Get library statistics."""
    conn = get_db()

    total_books = conn.execute("SELECT COUNT(*) FROM books WHERE ingestion_status='completed'").fetchone()[0]
    total_chunks = conn.execute("SELECT SUM(chunk_count) FROM books WHERE ingestion_status='completed'").fetchone()[0] or 0
    failed_books = conn.execute("SELECT COUNT(*) FROM books WHERE ingestion_status='failed'").fetchone()[0]
    categories = conn.execute(
        "SELECT category, COUNT(*) FROM books WHERE ingestion_status='completed' GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    recent = conn.execute(
        "SELECT title, category, date_added FROM books WHERE ingestion_status='completed' ORDER BY date_added DESC LIMIT 5"
    ).fetchall()
    conn.close()

    # Count inbox pending
    inbox_count = 0
    if os.path.isdir(INBOX_PATH):
        inbox_count = sum(1 for f in os.listdir(INBOX_PATH)
                         if Path(f).suffix.lower() in SUPPORTED_FORMATS)

    lines = [f"Book Library Status\n{'='*40}\n"]
    lines.append(f"Total books: {total_books}")
    lines.append(f"Total chunks: {total_chunks}")
    lines.append(f"Failed ingestions: {failed_books}")
    lines.append(f"Inbox pending: {inbox_count}")

    if categories:
        lines.append(f"\nCategories:")
        for cat, count in categories:
            lines.append(f"  {count:>3}  {cat}")

    if recent:
        lines.append(f"\nRecently added:")
        for title, cat, date in recent:
            lines.append(f"  - {title} [{cat}] ({date[:10]})")

    lines.append(f"\nPaths:")
    lines.append(f"  Database: {DB_PATH}")
    lines.append(f"  ChromaDB: {CHROMA_PATH}")
    lines.append(f"  Inbox: {INBOX_PATH}")
    lines.append(f"  Collections: {COLLECTIONS_PATH}")

    return [TextContent(type="text", text="\n".join(lines))]


# ---------------------------------------------------------------------------
# MCP Tool Definitions and Dispatch
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="ingest_books",
            description="Scan a folder for new ebooks, convert to text, chunk, embed with Gemini, "
                        "auto-categorize, and add to the library. Supports EPUB, MOBI, AZW3, PDF, TXT.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": f"Folder to scan for books. Defaults to inbox: {INBOX_PATH}"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Scan subdirectories too (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="search_books",
            description="Semantic search across the entire book library. "
                        "Finds passages by meaning, not just keywords.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (e.g., Trading, Philosophy, Technology)"
                    },
                    "book_title": {
                        "type": "string",
                        "description": "Filter to a specific book (partial match)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default 5, max 20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="ask_about_book",
            description="Find relevant passages from a specific book for a given question.",
            inputSchema={
                "type": "object",
                "properties": {
                    "book_title": {
                        "type": "string",
                        "description": "Book title (partial match supported)"
                    },
                    "question": {
                        "type": "string",
                        "description": "What you want to find in the book"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of passages to return (default 5, max 20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["book_title", "question"]
            }
        ),
        Tool(
            name="list_books",
            description="List all books in the library with metadata, grouped by category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by category"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_book_summary",
            description="Get the opening text (first 3 chunks) of a specific book.",
            inputSchema={
                "type": "object",
                "properties": {
                    "book_title": {
                        "type": "string",
                        "description": "Book title (partial match)"
                    }
                },
                "required": ["book_title"]
            }
        ),
        Tool(
            name="get_library_status",
            description="Get library statistics: total books, chunks, categories, inbox status.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls."""
    if name == "ingest_books":
        return await handle_ingest_books(arguments)
    elif name == "search_books":
        return await handle_search_books(arguments)
    elif name == "ask_about_book":
        return await handle_ask_about_book(arguments)
    elif name == "list_books":
        return await handle_list_books(arguments)
    elif name == "get_book_summary":
        return await handle_get_book_summary(arguments)
    elif name == "get_library_status":
        return await handle_get_library_status(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting Book Library MCP Server...")

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    try:
        init_db(DB_PATH)
        logger.info(f"Database initialized at {DB_PATH}")

        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP Server ready and listening on stdio")
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
