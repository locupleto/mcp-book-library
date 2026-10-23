==============================================================================
 BOOK LIBRARY — README
==============================================================================

A local semantic-search / Q&A library. Every book is converted to plain text,
chunked, embedded with Gemini, auto-categorized, and stored for search.

Current scale: ~7,760 books / ~151,500 chunks. About 95 are books; the rest
is the S&C Magazine article archive.

------------------------------------------------------------------------------
 HOW TO ADD A BOOK
------------------------------------------------------------------------------

Just drop the file into the inbox:

    /Volumes/Documents/BookLibrary/inbox/

That's it. A launchd folder-watcher auto-ingests it (convert -> chunk ->
embed -> categorize -> move into collections/<category>/). No manual step.
It takes a few minutes per book.

  * Supported formats: EPUB, MOBI, AZW3, AZW, PDF, TXT
  * Drop only ONE file per book. Dropping epub + mobi + pdf of the same title
    ingests it three times.
  * The file must be DRM-free. Kindle books bought from Amazon are not:
    see "KINDLE BOOKS YOU OWN" below.

WHICH FORMAT TO PREFER (when a book ships several):
  1. EPUB  -- BEST. Clean structured text, code listings stay intact.
  2. MOBI  -- OK fallback. Older/messier conversion.
  3. PDF   -- WORST. Fixed layout garbles code blocks, multi-column text,
              and adds header/footer noise. Avoid for code-heavy books.
Extraction quality is everything here, so favor EPUB.

TITLE AND AUTHOR:
  * EPUB / MOBI / AZW3 / AZW: taken from the metadata inside the book (what
    Calibre shows). "Ray Dalio" style is used, not "Dalio, Ray".
  * PDF / TXT, or an ebook without metadata: taken from the filename.
    Name the file    Title - Author.pdf
    The split is a guess (the shorter half is taken as the author), so a
    short title with a long author list can come out back to front.
  * Check after the ingest (see QUICK CHECKS). A wrong title or author has
    to be corrected in two places: the books table in SQLite and the chunk
    metadata (book_title, author) in ChromaDB. kindle-capture's library step
    does both automatically; for other books, ask Claude to do it.

------------------------------------------------------------------------------
 KINDLE BOOKS YOU OWN
------------------------------------------------------------------------------

Amazon Kindle books are DRM-protected and cannot be dropped in directly.
The kindle-capture tool reads an owned book off the screen of Kindle for Mac
instead (screenshots + Apple's on-device OCR), builds an EPUB, and hands it
to the library:

    kindle-capture capture --title MyBook        # Kindle open at the cover
    kindle-capture ocr     --title MyBook
    kindle-capture epub    --title MyBook --book-title "Full Title" --author "Author"
    kindle-capture library --title MyBook        # inbox -> ingest -> verified

The capture needs the Mac to itself for 5-15 minutes. Full instructions:
    /Volumes/Work/development/projects/git/kindle-capture/README.md
Output (private, never share): /Volumes/Documents/kindle-capture/<title>/

------------------------------------------------------------------------------
 HOW IT WORKS (machines & jobs)
------------------------------------------------------------------------------

INDEXING happens on MAC STUDIO (not Mac mini):
  * launchd agent: com.locupleto.book-library-ingest
    - WatchPath: /Volumes/Documents/BookLibrary/inbox  (auto-fires on new file)
    - Script:    ~/bin/book-library-ingest.sh
    - Python:    ~/.book-library/venv  (its OWN venv, separate from the repo's)
    - Logs:      ~/.book-library/ingest.log
                 ~/.book-library/launchd_{out,err}.log

  !! KEEP THE INGEST VENV IN STEP WITH THE CODE. The watcher runs the repo's
     current code on ~/.book-library/venv. When the code needs newer packages
     (requirements.txt), that venv must be upgraded too, or every ingest fails
     while the files sit in the inbox. This happened 2026-08-08 -> 2026-10-20
     (the MCP 2.x port; ingest.log showed a TypeError about on_list_tools).
     After changing requirements.txt:
         ~/.book-library/venv/bin/pip install -U -r \
             /Volumes/Work/development/projects/git/mcp-book-library/requirements.txt
     A file still sitting in the inbox minutes after the drop means a failed
     ingest: read the end of ingest.log.

NIGHTLY SYNC to Mac mini (crontab on Mac Studio, 03:30):
    /Volumes/Work/development/projects/git/mcp-book-library/sync_book_library.sh
  * Pushes the library to Mac mini (daily sync_YYYYMMDD.log files).
  * Mac mini holds a MIRROR only -- it does NO indexing.

  !! MIRROR LAG -- a book ingested during the day is live on Mac Studio
     immediately, but does NOT appear on the Mac mini mirror until the next
     03:30 sync. To publish it sooner, run the sync manually on Mac Studio:
         /Volumes/Work/development/projects/git/mcp-book-library/sync_book_library.sh
     Then RECONNECT the consumer's MCP server (/mcp -> book-library ->
     Reconnect, or restart its client) so it reloads the freshly-synced
     ChromaDB. A long-running MCP process caches the index in memory and will
     keep returning stale results -- or fail with "Error finding id" -- until
     it reconnects. This applies on Mac Studio too after any out-of-band
     change (manual purge/re-ingest), not just on the mirror.

------------------------------------------------------------------------------
 PATHS
------------------------------------------------------------------------------

  Inbox (drop here):  /Volumes/Documents/BookLibrary/inbox
  Collections:        /Volumes/Documents/BookLibrary/collections/<category>/
  SQLite DB:          ~/.book-library/book_library.db
  ChromaDB (vectors): ~/.book-library/chroma
  Ingest venv:        ~/.book-library/venv
  MCP server repo:    /Volumes/Work/development/projects/git/mcp-book-library
  kindle-capture:     /Volumes/Work/development/projects/git/kindle-capture

------------------------------------------------------------------------------
 QUICK CHECKS
------------------------------------------------------------------------------

  Library status:   book-library MCP -> get_library_status
  Ingest log:       tail -f ~/.book-library/ingest.log
  Inbox empty?      ls /Volumes/Documents/BookLibrary/inbox
                    (a file left there = a failed ingest)
  Latest books:     sqlite3 ~/.book-library/book_library.db \
                      "SELECT id, title, author, category, chunk_count,
                       ingestion_status FROM books ORDER BY id DESC LIMIT 5"
  Failed ingests:   sqlite3 ~/.book-library/book_library.db \
                      "SELECT id, title FROM books
                       WHERE ingestion_status != 'completed'"
==============================================================================
