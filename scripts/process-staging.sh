#!/bin/bash
set -euo pipefail

# Configuration via environment (set in CronJob manifest)
STAGING_DIR="${STAGING_DIR:-/data/audiobook-staging}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/media/books/audiobooks}"
STABILITY_SECONDS="${STABILITY_SECONDS:-300}"
BITRATE="${BITRATE:-64}"
WORKERS="${WORKERS:-1}"
STALE_MINUTES="${STALE_MINUTES:-60}"

# Audiobookshelf scan trigger (all three required to enable)
ABS_URL="${ABS_URL:-}"
ABS_API_KEY="${ABS_API_KEY:-}"
ABS_LIBRARY_ID="${ABS_LIBRARY_ID:-}"

# ntfy push notifications
NTFY_URL="${NTFY_URL:-}"
NTFY_TOPIC="${NTFY_TOPIC:-downloads}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

notify() {
    # Usage: notify "title" "message" "priority" "tags"
    [ -z "$NTFY_URL" ] && return
    curl -sf -X POST "$NTFY_URL/$NTFY_TOPIC" \
        -H "Title: $1" \
        -H "Priority: ${3:-default}" \
        -H "Tags: ${4:-books}" \
        -d "$2" >/dev/null 2>&1 || true
}

# Graceful shutdown: revert .processing to .complete on SIGTERM/SIGINT
CURRENT_PROCESSING=""
cleanup() {
    if [ -n "$CURRENT_PROCESSING" ] && [ -f "$CURRENT_PROCESSING/.processing" ]; then
        log "INTERRUPTED: reverting $(basename "$CURRENT_PROCESSING") to .complete"
        mv "$CURRENT_PROCESSING/.processing" "$CURRENT_PROCESSING/.complete"
        notify "Audiobook interrupted" "$(basename "$CURRENT_PROCESSING") - job killed" "default" "books,hourglass"
    fi
}
trap cleanup SIGTERM SIGINT

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
RECOVERED=0

for dir in "$STAGING_DIR"/*/; do
    [ -d "$dir" ] || continue

    book_name="$(basename "$dir")"

    # Skip already processed (silent -- these accumulate)
    if [ -f "$dir/.processed" ]; then
        continue
    fi

    # Skip explicitly excluded
    if [ -f "$dir/.skip" ]; then
        continue
    fi

    # Auto-recover stale .processing from killed CronJobs
    if [ -f "$dir/.processing" ]; then
        if [ "$(find "$dir/.processing" -mmin "+${STALE_MINUTES}" 2>/dev/null)" ]; then
            log "RECOVER $book_name (stale .processing after ${STALE_MINUTES}min, resetting)"
            mv "$dir/.processing" "$dir/.complete"
            RECOVERED=$((RECOVERED + 1))
        else
            log "SKIP $book_name (processing in progress)"
            SKIPPED=$((SKIPPED + 1))
        fi
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
    CURRENT_PROCESSING="$dir"

    # Run audiobook-optimizer (AI auto-enables if ANTHROPIC_API_KEY is set)
    if audiobook-optimizer process "$dir" "$OUTPUT_DIR" \
        --bitrate "$BITRATE" --workers "$WORKERS" 2>&1; then
        log "SUCCESS $book_name"
        rm -f "$dir/.processing"
        touch "$dir/.processed"
        PROCESSED=$((PROCESSED + 1))
        notify "Audiobook ready" "$book_name" "default" "books,white_check_mark"
    else
        EXIT_CODE=$?
        log "FAILED $book_name (exit code $EXIT_CODE)"
        mv "$dir/.processing" "$dir/.failed"
        FAILED=$((FAILED + 1))
        notify "Audiobook failed" "$book_name (exit $EXIT_CODE)" "high" "books,warning"
    fi
    CURRENT_PROCESSING=""
done

if [ $PROCESSED -eq 0 ] && [ $FAILED -eq 0 ] && [ $SKIPPED -eq 0 ] && [ $RECOVERED -eq 0 ]; then
    log "No audiobooks ready for processing."
else
    log "Summary: processed=$PROCESSED failed=$FAILED skipped=$SKIPPED recovered=$RECOVERED"
fi

# Trigger Audiobookshelf library scan after successful processing
if [ "$PROCESSED" -gt 0 ] && [ -n "$ABS_URL" ] && [ -n "$ABS_API_KEY" ] && [ -n "$ABS_LIBRARY_ID" ]; then
    log "Triggering Audiobookshelf library scan..."
    if curl -sf -X POST \
        -H "Authorization: Bearer $ABS_API_KEY" \
        "${ABS_URL}/api/libraries/${ABS_LIBRARY_ID}/scan" >/dev/null 2>&1; then
        log "ABS scan triggered"
    else
        log "WARN: ABS scan trigger failed (non-fatal)"
    fi
fi
