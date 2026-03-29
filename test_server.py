#!/usr/bin/env python3
"""
Tests for Book Library MCP Server

Unit tests that run without Gemini API access (except integration tests).

Groups:
1. Text chunking
2. File hash calculation
3. Filename metadata extraction
4. Database operations
5. Tool validation
6. Integration: ingest + search (requires GEMINI_API_KEY)
"""

import hashlib
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from book_library_mcp_server import (
    split_text_into_chunks,
    calculate_file_hash,
    extract_metadata_from_filename,
    init_db,
    SUPPORTED_FORMATS,
    VALID_CATEGORIES,
)

passed = 0
failed = 0
errors = []


def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(name)


# =========================================================================
# Group 1: Text Chunking
# =========================================================================

def test_text_chunking():
    print("\n--- Group 1: Text Chunking ---")

    # Short text stays as one chunk
    short = "This is a short text."
    chunks = split_text_into_chunks(short, chunk_size=1500, overlap=200)
    test("Short text = single chunk",
         len(chunks) == 1 and chunks[0] == short)

    # Empty text returns empty list
    test("Empty text returns empty list",
         split_text_into_chunks("") == [])

    test("Whitespace-only returns empty list",
         split_text_into_chunks("   \n\n  ") == [])

    # Long text gets split
    long_text = "Word " * 500  # ~2500 chars
    chunks = split_text_into_chunks(long_text, chunk_size=500, overlap=50)
    test("Long text splits into multiple chunks",
         len(chunks) > 1,
         f"got {len(chunks)} chunks")

    # All text is represented (accounting for overlap, all words should appear)
    rejoined = " ".join(chunks)
    test("All content preserved in chunks",
         rejoined.count("Word") >= 400,
         f"found {rejoined.count('Word')} occurrences")

    # Chunks respect size limit (approximately)
    test("Chunks respect size limit",
         all(len(c) <= 600 for c in chunks),  # some tolerance
         f"max chunk size: {max(len(c) for c in chunks)}")

    # Paragraph boundaries preferred
    para_text = "First paragraph about topic A.\n\nSecond paragraph about topic B.\n\nThird paragraph about topic C."
    chunks = split_text_into_chunks(para_text, chunk_size=60, overlap=10)
    test("Paragraphs split on double newline",
         len(chunks) >= 2,
         f"got {len(chunks)} chunks: {chunks}")


# =========================================================================
# Group 2: File Hash
# =========================================================================

def test_file_hash():
    print("\n--- Group 2: File Hash ---")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content for hashing")
        path = f.name

    try:
        h = calculate_file_hash(path)
        test("Hash returns hex string",
             isinstance(h, str) and len(h) == 64)

        # Same content = same hash
        h2 = calculate_file_hash(path)
        test("Same file = same hash",
             h == h2)

        # Different content = different hash
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("different content")
            path2 = f2.name

        h3 = calculate_file_hash(path2)
        test("Different file = different hash",
             h != h3)
        os.unlink(path2)

    finally:
        os.unlink(path)


# =========================================================================
# Group 3: Filename Metadata Extraction
# =========================================================================

def test_filename_extraction():
    print("\n--- Group 3: Filename Metadata Extraction ---")

    # Title - Author format
    title, author = extract_metadata_from_filename("/books/How to Day Trade - Andrew Aziz.pdf")
    test("Title-Author split",
         "Day Trade" in title and "Andrew Aziz" in author,
         f"got title='{title}', author='{author}'")

    # PDFDrive artifact removal
    title, author = extract_metadata_from_filename("/books/Python Cookbook ( PDFDrive ).pdf")
    test("PDFDrive artifact stripped",
         "PDFDrive" not in title,
         f"got title='{title}'")

    # Underscore replacement
    title, author = extract_metadata_from_filename("/books/the_art_of_war.epub")
    test("Underscores replaced with spaces",
         "_" not in title,
         f"got title='{title}'")

    # No separator = whole name as title
    title, author = extract_metadata_from_filename("/books/Thinking Fast and Slow.mobi")
    test("No separator: title is full name, author Unknown",
         "Thinking Fast and Slow" in title and author == "Unknown",
         f"got title='{title}', author='{author}'")

    # Complex messy filename
    title, author = extract_metadata_from_filename(
        "/books/How to Day Trade for a Living_ A Beginners Guide ( PDFDrive ).pdf"
    )
    test("Complex filename cleaned up",
         "PDFDrive" not in title and len(title) > 10,
         f"got title='{title}'")

    # book-drive.com artifact
    title, author = extract_metadata_from_filename("/books/Some Book (book-drive.com).epub")
    test("book-drive.com artifact stripped",
         "book-drive" not in title,
         f"got title='{title}'")


# =========================================================================
# Group 4: Database Operations
# =========================================================================

def test_database_operations():
    print("\n--- Group 4: Database Operations ---")

    conn = init_db(":memory:")

    # Tables exist
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [t[0] for t in tables]
    test("All tables created",
         "books" in table_names and "book_texts" in table_names,
         f"got {table_names}")

    # Insert a book
    cursor = conn.execute(
        """INSERT INTO books (title, author, file_path, original_format, file_hash,
           file_size_bytes, ingestion_status) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("Test Book", "Test Author", "/path/test.pdf", "pdf", "abc123hash", 1024, "completed")
    )
    book_id = cursor.lastrowid
    conn.commit()
    test("Book inserted successfully",
         book_id is not None and book_id > 0)

    # Duplicate hash rejected
    try:
        conn.execute(
            """INSERT INTO books (title, author, file_path, original_format, file_hash,
               file_size_bytes, ingestion_status) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("Duplicate", "Author", "/path/dup.pdf", "pdf", "abc123hash", 1024, "completed")
        )
        conn.commit()
        test("Duplicate hash rejected", False, "should have raised IntegrityError")
    except sqlite3.IntegrityError:
        test("Duplicate hash rejected", True)

    # Insert book text
    conn.execute(
        "INSERT INTO book_texts (book_id, full_text, text_length) VALUES (?, ?, ?)",
        (book_id, "Full text of the book here.", 27)
    )
    conn.commit()
    text_row = conn.execute("SELECT full_text FROM book_texts WHERE book_id=?", (book_id,)).fetchone()
    test("Book text stored and retrieved",
         text_row is not None and "Full text" in text_row[0])

    # Category query
    conn.execute(
        """INSERT INTO books (title, author, file_path, original_format, file_hash,
           file_size_bytes, category, ingestion_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("Trading Book", "Trader", "/path/trade.pdf", "pdf", "def456hash", 2048, "Trading", "completed")
    )
    conn.commit()
    trading = conn.execute("SELECT COUNT(*) FROM books WHERE category='Trading'").fetchone()[0]
    test("Category filter works",
         trading == 1)

    conn.close()


# =========================================================================
# Group 5: Tool Validation
# =========================================================================

def test_tool_validation():
    print("\n--- Group 5: Tool Validation ---")

    import asyncio
    import book_library_mcp_server as srv

    # Use temp DB
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db = f.name

    original_db = srv.DB_PATH
    srv.DB_PATH = temp_db
    init_db(temp_db)

    try:
        # search_books with empty query
        result = asyncio.run(srv.handle_search_books({"query": ""}))
        test("search_books with empty query returns error",
             "Error:" in result[0].text)

        # ask_about_book missing title
        result = asyncio.run(srv.handle_ask_about_book({"question": "test"}))
        test("ask_about_book missing title returns error",
             "Error:" in result[0].text)

        # ask_about_book missing question
        result = asyncio.run(srv.handle_ask_about_book({"book_title": "test"}))
        test("ask_about_book missing question returns error",
             "Error:" in result[0].text)

        # get_book_summary missing title
        result = asyncio.run(srv.handle_get_book_summary({"book_title": ""}))
        test("get_book_summary missing title returns error",
             "Error:" in result[0].text)

        # ingest_books with nonexistent folder
        result = asyncio.run(srv.handle_ingest_books({"folder_path": "/nonexistent/folder"}))
        test("ingest_books with bad folder returns error",
             "Error:" in result[0].text)

        # list_books on empty library
        result = asyncio.run(srv.handle_list_books({}))
        test("list_books on empty library",
             "empty" in result[0].text.lower())

        # get_library_status on empty library
        result = asyncio.run(srv.handle_get_library_status({}))
        test("get_library_status works on empty library",
             "Total books: 0" in result[0].text)

        # Tool list
        tools = asyncio.run(srv.list_tools())
        tool_names = [t.name for t in tools]
        test("All 6 tools listed",
             len(tool_names) == 6,
             f"got {len(tool_names)}: {tool_names}")

        test("Expected tools present",
             all(t in tool_names for t in ["ingest_books", "search_books", "ask_about_book",
                                            "list_books", "get_book_summary", "get_library_status"]))

    finally:
        os.unlink(temp_db)
        srv.DB_PATH = original_db


# =========================================================================
# Group 6: Constants and Configuration
# =========================================================================

def test_configuration():
    print("\n--- Group 6: Configuration ---")

    test("Supported formats include common ebook types",
         all(f in SUPPORTED_FORMATS for f in ['.epub', '.mobi', '.pdf', '.txt']))

    test("Valid categories is non-empty",
         len(VALID_CATEGORIES) > 10)

    test("Other is a valid category (fallback)",
         "Other" in VALID_CATEGORIES)


# =========================================================================
# Run All Tests
# =========================================================================

def main():
    print("=" * 60)
    print("Book Library MCP Server — Test Suite")
    print("=" * 60)

    test_text_chunking()
    test_file_hash()
    test_filename_extraction()
    test_database_operations()
    test_tool_validation()
    test_configuration()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"Failed tests: {', '.join(errors)}")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
