#!/bin/bash
DATABASE=${1:-tpch}
DATAPATH=${2:-data/olap/tpch}
STRATEGY=${3:-smac}
N_PROPOSALS=${4:-100}

echo "=== Phase 1 Sampling: ${DATABASE} / ${STRATEGY} / ${N_PROPOSALS} proposals ==="
python main.py \
  --config config/cloud.ini \
  --database "${DATABASE}" \
  --datapath "${DATAPATH}" \
  --strategy "${STRATEGY}" \
  --n-proposals "${N_PROPOSALS}" \
  --resume

echo "=== Done. Backup offline_sample ==="
mkdir -p "offline_sample/${DATABASE}/backup"
cp "offline_sample/${DATABASE}/offline_sample_${DATABASE}_localhost.jsonl" \
   "offline_sample/${DATABASE}/backup/offline_sample_${DATABASE}_$(date +%Y%m%d_%H%M).jsonl"