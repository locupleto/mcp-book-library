#!/bin/bash
# sync_book_library.sh — Nightly cron job on Mac Studio
# Zips ChromaDB + SQLite, SCPs to Mac mini, unpacks there
#
# Crontab entry (Mac Studio):
#   30 3 * * * /Volumes/Work/development/projects/git/mcp-book-library/sync_book_library.sh

LOG_DIR="$HOME/.book-library"
LOG_FILE="$LOG_DIR/sync_$(date +%Y%m%d).log"
DB_PATH="$HOME/.book-library/book_library.db"
CHROMA_PATH="$HOME/.book-library/chroma"
TMP_ARCHIVE="/tmp/book_library_sync.tar.gz"

# Mac mini target (via Tailscale)
MACMINI_USER="urban"
MACMINI_HOST="100.102.226.1"
MACMINI_KEY="$HOME/.ssh/macmini"
MACMINI_DEST="$HOME/.book-library/"

mkdir -p "$LOG_DIR"

echo "=== Book Library Sync $(date) ===" >> "$LOG_FILE"

# Check if there's anything to sync
if [ ! -f "$DB_PATH" ]; then
    echo "No database found at $DB_PATH. Nothing to sync." >> "$LOG_FILE"
    exit 0
fi

# Create tar.gz of both SQLite and ChromaDB
echo "Creating archive..." >> "$LOG_FILE"
tar -czf "$TMP_ARCHIVE" -C "$HOME/.book-library" book_library.db chroma 2>> "$LOG_FILE"
TAR_EXIT=$?

if [ $TAR_EXIT -ne 0 ]; then
    echo "ERROR: tar failed (exit $TAR_EXIT)" >> "$LOG_FILE"
    exit 1
fi

ARCHIVE_SIZE=$(du -h "$TMP_ARCHIVE" | cut -f1)
echo "Archive created: $ARCHIVE_SIZE" >> "$LOG_FILE"

# SCP to Mac mini
echo "SCPing to Mac mini..." >> "$LOG_FILE"
scp -i "$MACMINI_KEY" "$TMP_ARCHIVE" "$MACMINI_USER@$MACMINI_HOST:/tmp/book_library_sync.tar.gz" >> "$LOG_FILE" 2>&1
SCP_EXIT=$?

if [ $SCP_EXIT -ne 0 ]; then
    echo "ERROR: SCP failed (exit $SCP_EXIT)" >> "$LOG_FILE"
    rm -f "$TMP_ARCHIVE"
    exit 1
fi

# Unpack on Mac mini (replace existing)
echo "Unpacking on Mac mini..." >> "$LOG_FILE"
ssh -i "$MACMINI_KEY" "$MACMINI_USER@$MACMINI_HOST" "
    mkdir -p ~/.book-library
    cd ~/.book-library
    rm -rf chroma book_library.db
    tar -xzf /tmp/book_library_sync.tar.gz
    rm -f /tmp/book_library_sync.tar.gz
    echo \"Unpacked: \$(sqlite3 book_library.db 'SELECT COUNT(*) FROM books WHERE ingestion_status=\"completed\"') books\"
" >> "$LOG_FILE" 2>&1
SSH_EXIT=$?

rm -f "$TMP_ARCHIVE"

if [ $SSH_EXIT -eq 0 ]; then
    echo "Sync completed successfully." >> "$LOG_FILE"
else
    echo "ERROR: Unpack on Mac mini failed (exit $SSH_EXIT)" >> "$LOG_FILE"
fi

echo "=== Done $(date) ===" >> "$LOG_FILE"
