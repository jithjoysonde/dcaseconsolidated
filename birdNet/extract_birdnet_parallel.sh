#!/bin/bash
# Parallel BirdNet Extraction Script - Optimized for Multi-core CPU
# This script runs multiple extraction processes in parallel to maximize CPU usage

set -e

source /data/4joyson/dcase_t5/bin/activate
cd /data/4joyson/bdnt/birdnet

echo "=========================================="
echo "Parallel BirdNet Extraction (GPU-Optimized)"
echo "=========================================="
echo "Configuration:"
echo "  - Window size: 3s"
echo "  - Hop size: 3s (non-overlapping)"
echo "  - Output dim: 6522"
echo "  - Threads per process: 8"
echo "  - Parallel processes: 5 (one per class)"
echo ""

# Function to extract a single class directory
extract_class() {
    local class_dir=$1
    local dataset=$2
    echo "[$(date +%H:%M:%S)] Starting extraction: $dataset/$class_dir"
    python extract_embeddings.py \
        --wav-dir "/data/msc-proj/$dataset/$class_dir" \
        --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
        --save-in-place \
        2>&1 | grep -E "Saved|it/s|%|skip" || true
    echo "[$(date +%H:%M:%S)] ✓ Completed: $dataset/$class_dir"
}

# Export function for parallel execution
export -f extract_class

echo "=========================================="
echo "Phase 1: Training Set (5 classes)"
echo "=========================================="
echo ""

# Extract Training Set classes in parallel (5 processes)
echo "Starting parallel extraction of Training_Set..."
parallel -j 5 extract_class {} Training_Set ::: HT JD BV WMW MT

echo ""
echo "=========================================="
echo "Phase 2: Validation Set (3 classes)"
echo "=========================================="
echo ""

# Extract Validation Set classes in parallel (3 processes)
echo "Starting parallel extraction of Validation_Set..."
parallel -j 3 extract_class {} Validation_Set_DSAI_2025_2026 ::: ME PB PB24

echo ""
echo "=========================================="
echo "Extraction Complete!"
echo "=========================================="
echo "All embeddings saved as *_BDnet.npy"
echo "Shape: (num_windows, 6522) per file"
echo ""
echo "Next: python train.py --config-name train_birdnet +exp_name='BirdNet'"
echo "=========================================="
