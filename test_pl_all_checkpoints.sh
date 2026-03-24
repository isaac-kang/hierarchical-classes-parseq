#!/bin/bash
# Test all checkpoints in a given directory using test_pl.py, then summarize results.
# Usage: ./test_pl_all_checkpoints.sh <checkpoint_dir> [extra_args...]
# Example: ./test_pl_all_checkpoints.sh outputs/Quokka/train_pl-on-real/2026-03-23_09-21-11/checkpoints --cased

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

# Average checkpoints first
echo "========================================"
echo "Averaging checkpoints..."
echo "========================================"
python average_checkpoints.py "$CKPT_DIR"
echo ""

for ckpt in $CKPTS; do
    echo "========================================"
    echo "Testing: $ckpt"
    echo "========================================"
    python test_pl.py "$ckpt" $EXTRA_ARGS
    echo ""
done

# Test averaged checkpoint if it was created
AVG_CKPT="$CKPT_DIR/top3_wa.ckpt"
if [ -f "$AVG_CKPT" ]; then
    echo "========================================"
    echo "Testing: $AVG_CKPT (weight averaged)"
    echo "========================================"
    python test_pl.py "$AVG_CKPT" $EXTRA_ARGS
    echo ""
fi

# ── Summary ──
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"

# Extract experiment spec from checkpoint path (strip leading outputs/ and trailing /checkpoints)
EXPERIMENT=$(echo "$CKPT_DIR" | sed 's|.*outputs/||' | sed 's|/checkpoints/*$||')
echo "Experiment: $EXPERIMENT"
echo ""

# Header
printf "%-60s  %10s  %10s\n" "Checkpoint" "val_acc" "bench_sub"
printf "%-60s  %10s  %10s\n" "------------------------------------------------------------" "----------" "----------"

for log in $(find "$CKPT_DIR" -name "*.ckpt.log.txt" | sort); do
    ckpt_name=$(basename "$log" .log.txt)

    # Extract val_accuracy from checkpoint filename (e.g. val_accuracy=94.3042)
    val_acc=$(echo "$ckpt_name" | grep -oP 'val_accuracy=\K[0-9.]+' || echo "-")

    # Extract Benchmark (Subset) Combined accuracy from log file
    bench_sub=$(awk '/^Benchmark \(Subset\) set:/{found=1} found && /\| Combined/{print $6; exit}' "$log")
    bench_sub=${bench_sub:-"-"}

    printf "%-60s  %10s  %10s\n" "$ckpt_name" "$val_acc" "$bench_sub"
done
