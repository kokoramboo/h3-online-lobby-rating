#!/bin/bash

# H3 Online Lobby Rating System - Pipeline Wrapper
# This script automates the daily crawl and the full per-group rating update.

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/pipeline.log"
mkdir -p "$LOG_DIR"

echo "================================================================================" >> "$LOG_FILE"
echo "Starting Ranking Pipeline: $(date)" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"

# 1. Run Orchestrated Rating Pipeline
# This handles:
# - Match filtering per template group
# - Automatic resume/priors check
# - Granular TAU calibration (Triple-Key: Template, Size, Random)
# - Bayesian Skill Inference (Gaussian Skill Rating with Dynamic BETA)
# - Analytical LCB/WinRate computation
echo "[*] Running Rating Pipeline (Orchestrator)..." | tee -a "$LOG_FILE"
python3 -u rating_pipeline.py 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[!] Rating pipeline failed. Check logs." | tee -a "$LOG_FILE"
    exit 1
fi

# 2. Validate Health
echo "[*] Validating model health..." | tee -a "$LOG_FILE"
python3 -u validate_ratings.py --elite-only 2>&1 | tee -a "$LOG_FILE"

echo "================================================================================" >> "$LOG_FILE"
echo "Pipeline Completed Successfully: $(date)" >> "$LOG_FILE"
echo "Leaderboards Updated in data/groups/" >> "$LOG_FILE"
echo "================================================================================" >> "$LOG_FILE"

echo "[*] Pipeline complete. See $LOG_FILE for details."
