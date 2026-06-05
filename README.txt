==============================================================================
 BOOK LIBRARY — README
==============================================================================

A local semantic-search / Q&A library. Every book is converted to plain text,
chunked, embedded with Gemini, auto-categorized, and stored for search.

Current scale: ~7,700 books / ~145,000 chunks (mostly S&C Magazine archive).

------------------------------------------------------------------------------
 HOW TO ADD A BOOK
------------------------------------------------------------------------------

Just drop the file into the inbox:

    /Volumes/Documents/BookLibrary/inbox/

That's it. A launchd folder-watcher auto-ingests it (convert -> chunk ->
embed -> categorize -> move into collections/<category>/). No manual step.

  * Supported formats: EPUB, MOBI, AZW3, AZW, PDF, TXT
  * Drop only ONE file per book. Dropping epub + mobi + pdf of the same title
    ingests it three times.

WHICH FORMAT TO PREFER (when a book ships several):
  1. EPUB  -- BEST. Clean structured text, code listings stay intact.
  2. MOBI  -- OK fallback. Older/messier conversion.
  3. PDF   -- WORST. Fixed layout garbles code blocks, multi-column text,
              and adds header/footer noise. Avoid for code-heavy books.
Extraction quality is everything here, so favor EPUB.

------------------------------------------------------------------------------
 HOW IT WORKS (machines & jobs)
------------------------------------------------------------------------------

INDEXING happens on MAC STUDIO (not Mac mini):
  * launchd agent: com.locupleto.book-library-ingest
    - WatchPath: /Volumes/Documents/BookLibrary/inbox  (auto-fires on new file)
    - Script:    ~/bin/book-library-ingest.sh
    - Logs:      ~/.book-library/ingest.log
                 ~/.book-library/launchd_{out,err}.log

NIGHTLY SYNC to Mac mini (crontab on Mac Studio, 03:30):
    /Volumes/Work/development/projects/git/mcp-book-library/sync_book_library.sh
  * Pushes the library to Mac mini (daily sync_YYYYMMDD.log files).
  * Mac mini holds a MIRROR only -- it does NO indexing.

------------------------------------------------------------------------------
 PATHS
------------------------------------------------------------------------------

  Inbox (drop here):  /Volumes/Documents/BookLibrary/inbox
  Collections:        /Volumes/Documents/BookLibrary/collections/<category>/
  SQLite DB:          ~/.book-library/book_library.db
  ChromaDB (vectors): ~/.book-library/chroma
  MCP server repo:    /Volumes/Work/development/projects/git/mcp-book-library

------------------------------------------------------------------------------
 QUICK CHECKS
------------------------------------------------------------------------------

  Library status:   book-library MCP -> get_library_status
  Ingest log:       tail -f ~/.book-library/ingest.log
==============================================================================
