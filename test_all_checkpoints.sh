#!/bin/bash
# Test all checkpoints in a given directory using test_pl.py
# Usage: ./test_all_checkpoints.sh <checkpoint_dir> [extra_args...]
# Example: ./test_all_checkpoints.sh outputs/base-group/base-name/2026-03-22_01-00-00/checkpoints --cased

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_dir> [extra_args...]"
    exit 1
fi

CKPT_DIR="$1"
shift
EXTRA_ARGS="$@"

if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Directory '$CKPT_DIR' not found"
    exit 1
fi

CKPTS=$(find "$CKPT_DIR" -name "*.ckpt" | sort)

if [ -z "$CKPTS" ]; then
    echo "No .ckpt files found in $CKPT_DIR"
    exit 1
fi

echo "Found checkpoints:"
echo "$CKPTS"
echo ""

for ckpt in $CKPTS; do
    echo "========================================"
    echo "Testing: $ckpt"
    echo "========================================"
    python test_pl.py "$ckpt" $EXTRA_ARGS
    echo ""
done
