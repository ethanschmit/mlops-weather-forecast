#!/bin/bash
# pipeline.sh
# Runs the full retraining pipeline: ingest → features → train → evaluate
# Called by GitHub Actions on a schedule, or manually with: bash pipeline.sh
#
# set -e means: if any command returns a non-zero exit code, stop immediately
# Without this, a failed ingest would silently continue to train on stale data
set -e
set -o pipefail    # catch errors inside pipes too

echo "======================================"
echo "PIPELINE START: $(date)"
echo "======================================"

echo ""
echo "[1/4] Ingesting data..."
python src/ingest.py

echo ""
echo "[2/4] Building features..."
python src/features.py

echo ""
echo "[3/4] Training model..."
# Capture the output line that contains RUN_ID and CV_MAE
TRAIN_OUTPUT=$(python src/train.py)
echo "$TRAIN_OUTPUT"

# Parse RUN_ID and MAE from the output
# train.py prints exactly: RUN_ID=abc123 CV_MAE=1.823456
RUN_ID=$(echo "$TRAIN_OUTPUT" | grep -oP 'RUN_ID=\K\S+')
MAE=$(echo "$TRAIN_OUTPUT"    | grep -oP 'CV_MAE=\K\S+')

echo ""
echo "[4/4] Evaluating and promoting (run=$RUN_ID, mae=$MAE)..."
python src/evaluate.py --run_id "$RUN_ID" --mae "$MAE"

echo ""
echo "======================================"
echo "PIPELINE COMPLETE: $(date)"
echo "======================================"