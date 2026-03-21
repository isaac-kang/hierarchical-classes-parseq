#!/usr/bin/env bash
# Generate Pseudo-Labels for all LMDB datasets under
# /data/isaackang/data/STR/parseq/english/lmdb/train/real
# Discovers leaf LMDB dirs (those containing data.mdb) at any depth.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

CHECKPOINT="pretrained=parseq"
DATA_ROOT="/data/isaackang/data/STR/parseq/english/lmdb"
SEARCH_ROOT="${DATA_ROOT}/train/real"
OUTPUT_DIR="confusion_pl_output"

# Find all dirs that contain data.mdb and compute their path relative to DATA_ROOT
PL_DATASETS=()
while IFS= read -r mdb_file; do
    lmdb_dir=$(dirname "$mdb_file")
    rel_path="${lmdb_dir#${DATA_ROOT}/}"
    PL_DATASETS+=("${rel_path}")
done < <(find "${SEARCH_ROOT}" -name "data.mdb" | sort)

if [ ${#PL_DATASETS[@]} -eq 0 ]; then
    echo "No LMDB datasets found under ${SEARCH_ROOT}"
    exit 1
fi

echo "Found ${#PL_DATASETS[@]} LMDB datasets:"
for ds in "${PL_DATASETS[@]}"; do
    echo "  ${ds}"
done
echo ""

python confusion_and_pl.py \
    --checkpoint "${CHECKPOINT}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --pl_datasets "${PL_DATASETS[@]}"
