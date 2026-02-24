#!/bin/bash
set -euo pipefail

# Configuration via environment (set in CronJob manifest)
STAGING_DIR="${STAGING_DIR:-/data/audiobook-staging}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/media/books/audiobooks}"
STABILITY_SECONDS="${STABILITY_SECONDS:-300}"
BITRATE="${BITRATE:-64}"
WORKERS="${WORKERS:-1}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Exit early if staging dir doesn't exist or is empty
if [ ! -d "$STAGING_DIR" ]; then
    log "Staging directory $STAGING_DIR does not exist. Nothing to do."
    exit 0
fi

# Ensure output dir exists
mkdir -p "$OUTPUT_DIR"

PROCESSED=0
FAILED=0
SKIPPED=0

for dir in "$STAGING_DIR"/*/; do
    [ -d "$dir" ] || continue

    book_name="$(basename "$dir")"

    # Skip if already processing (stale from killed CronJob)
    if [ -f "$dir/.processing" ]; then
        log "SKIP $book_name (already processing -- stale?)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if previously failed
    if [ -f "$dir/.failed" ]; then
        log "SKIP $book_name (previously failed -- needs manual intervention)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Gate 1: .complete marker must exist
    if [ ! -f "$dir/.complete" ]; then
        continue
    fi

    # Gate 2: mtime stability -- no non-dotfile modified in last N seconds
    STABILITY_MINUTES=$((STABILITY_SECONDS / 60))
    if [ "$STABILITY_MINUTES" -lt 1 ]; then
        STABILITY_MINUTES=1
    fi
    RECENT_FILES=$(find "$dir" -type f -not -name '.*' -mmin "-${STABILITY_MINUTES}" 2>/dev/null | head -1)
    if [ -n "$RECENT_FILES" ]; then
        log "WAIT $book_name (files still being modified)"
        continue
    fi

    log "PROCESSING $book_name"

    # Atomic state transition
    mv "$dir/.complete" "$dir/.processing"

    # Run audiobook-optimizer (AI auto-enables if ANTHROPIC_API_KEY is set)
    if audiobook-optimizer process "$dir" "$OUTPUT_DIR" \
        --bitrate "$BITRATE" --workers "$WORKERS" 2>&1; then
        log "SUCCESS $book_name"
        rm -rf "$dir"
        PROCESSED=$((PROCESSED + 1))
    else
        EXIT_CODE=$?
        log "FAILED $book_name (exit code $EXIT_CODE)"
        mv "$dir/.processing" "$dir/.failed"
        FAILED=$((FAILED + 1))
    fi
done

if [ $PROCESSED -eq 0 ] && [ $FAILED -eq 0 ] && [ $SKIPPED -eq 0 ]; then
    log "No audiobooks ready for processing."
else
    log "Summary: processed=$PROCESSED failed=$FAILED skipped=$SKIPPED"
fi
