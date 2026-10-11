#!/bin/sh
set -e

# docker/entrypoint.sh: Ensures persistent mount paths exist and are owned by fmops
# before dropping privileges to run uvicorn.

ARTIFACTS_DIR="${FMOPS_PATHS__ARTIFACTS_DIR:-/app/artifacts}"
DATA_DIR="${FMOPS_PATHS__DATA_DIR:-/app/data}"

# If running as root (e.g. initial container start with mounted volumes):
if [ "$(id -u)" = "0" ]; then
    # Ensure standard artifacts subdirectories exist
    mkdir -p "$ARTIFACTS_DIR/models" "$ARTIFACTS_DIR/reports" "$ARTIFACTS_DIR/state"
    mkdir -p "$DATA_DIR/raw" "$DATA_DIR/processed" "$DATA_DIR/sample" "$DATA_DIR/reference"

    # Fix ownership of the mounted volumes so user fmops (UID 10001) has full read/write
    chown -R fmops:fmops "$ARTIFACTS_DIR" "$DATA_DIR"

    # Drop privileges cleanly to fmops and execute the CMD
    exec gosu fmops "$@"
fi

# If already running as non-root (e.g. in environments enforcing USER fmops directly):
exec "$@"
