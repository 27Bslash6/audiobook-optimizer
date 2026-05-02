#!/bin/bash
set -euo pipefail

# on-download-complete.sh — Touch .complete marker when a download client
# finishes an audiobook download. Designed to be called by:
#
#   qBittorrent:  Settings → Downloads → Run external program on torrent finished
#                 Command:  /scripts/on-download-complete.sh "%L" "%R"
#                 (%L = category, %R = content root path)
#
#   SABnzbd:      Config → Categories → audiobooks → Script: on-download-complete.sh
#                 SABnzbd passes args positionally:
#                   $1=final_dir, $2=nzb_name, ..., $8=category
#
#   NZBGet:       Settings → Category "audiobooks" → Post-Script: on-download-complete.sh
#                 NZBGet uses env vars: NZBPP_CATEGORY, NZBPP_DIRECTORY
#
# The script must be mounted/accessible inside the download client's container.
# The audiobook-staging volume must also be mounted in the client pod.

CATEGORY_MATCH="audiobooks"

log() { echo "[on-download-complete] $*"; }

# --- Detect calling convention ---

if [ -n "${NZBPP_CATEGORY:-}" ]; then
    # NZBGet: env var convention
    CATEGORY="$NZBPP_CATEGORY"
    CONTENT_PATH="$NZBPP_DIRECTORY"
elif [ $# -ge 8 ] && [ -d "$1" ]; then
    # SABnzbd: positional args ($1=dir, $8=category)
    CONTENT_PATH="$1"
    CATEGORY="${8:-}"
elif [ $# -ge 2 ]; then
    # qBittorrent: $1=category, $2=content_path
    CATEGORY="$1"
    CONTENT_PATH="$2"
else
    log "ERROR: unrecognized calling convention (args: $*)"
    exit 1
fi

# --- Gate: only act on audiobooks category ---

if [ "$CATEGORY" != "$CATEGORY_MATCH" ]; then
    exit 0
fi

# --- Validate path ---

if [ ! -d "$CONTENT_PATH" ]; then
    # Single-file torrent — content path is a file, not a directory.
    # The optimizer expects directories, so we can't process this directly.
    log "WARN: $CONTENT_PATH is not a directory (single-file download). Skipping .complete marker."
    exit 0
fi

# --- Mark complete ---

touch "$CONTENT_PATH/.complete"
log "Marked complete: $CONTENT_PATH"
