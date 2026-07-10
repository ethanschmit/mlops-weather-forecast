#!/bin/bash
# pipeline.sh
# Runs the full retraining pipeline: ingest → features → train → evaluate → drift
# Called by GitHub Actions on a schedule, or manually with: bash pipeline.sh
#
# set -e means: if any command returns a non-zero exit code, stop immediately.
# Without this, a failed ingest would silently continue to train on stale data.
set -e
set -o pipefail    # catch errors inside pipes too

echo "======================================"
echo "PIPELINE START: $(date)"
echo "======================================"

echo ""
echo "[1/5] Ingesting data..."
python src/ingest.py

echo ""
echo "[2/5] Building features..."
python src/features.py

echo ""
echo "[3/5] Training model..."
TRAIN_OUTPUT=$(python src/train.py)
echo "$TRAIN_OUTPUT"

RUN_ID=$(echo "$TRAIN_OUTPUT" | grep -oP 'RUN_ID=\K\S+')
MAE=$(echo "$TRAIN_OUTPUT"    | grep -oP 'CV_MAE=\K\S+')

echo ""
echo "[4/5] Evaluating and promoting (run=$RUN_ID, mae=$MAE)..."
# evaluate.py exits 1 when it rejects the new model — that's a valid,
# expected outcome, not a pipeline failure. The if/else here stops that
# exit code from tripping `set -e` and failing the whole run.
if python src/evaluate.py --run_id "$RUN_ID" --mae "$MAE"; then
    echo "Result: model promoted"
else
    echo "Result: new model rejected — champion unchanged"
fi

echo ""
echo "[5/5] Generating drift report..."
python src/drift_report.py

echo ""
echo "======================================"
echo "PIPELINE COMPLETE: $(date)"
echo "======================================"