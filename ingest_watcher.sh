#!/bin/bash
# ingest_watcher.sh — Triggered by launchd when files appear in BookInbox
# Waits a few seconds for file copy to complete, then ingests new books

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/python3"
LOG_FILE="$HOME/.book-library/ingest.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "=== Ingest triggered $(date) ===" >> "$LOG_FILE"

# Wait for file copy to finish (large EPUBs/PDFs)
sleep 3

GEMINI_API_KEY="${GEMINI_API_KEY}" \
BOOK_LIBRARY_INBOX="${BOOK_LIBRARY_INBOX:-/Volumes/Documents/BookInbox}" \
CALIBRE_EBOOK_CONVERT="${CALIBRE_EBOOK_CONVERT:-/opt/homebrew/bin/ebook-convert}" \
$VENV -c "
import asyncio
from book_library_mcp_server import handle_ingest_books, init_db, DB_PATH
init_db(DB_PATH)
result = asyncio.run(handle_ingest_books({}))
for r in result:
    print(r.text)
" >> "$LOG_FILE" 2>&1

echo "=== Done $(date) ===" >> "$LOG_FILE"
