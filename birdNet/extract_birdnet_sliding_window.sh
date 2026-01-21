#!/bin/bash
# Script to extract BirdNet embeddings using sliding windows (3s non-overlapping)
# This creates temporal embeddings: (num_windows, 6522) for each audio file

set -e

source /data/4joyson/dcase_t5/bin/activate

cd /data/4joyson/bdnt/birdnet

echo "=========================================="
echo "BirdNet Sliding Window Extraction"
echo "=========================================="
echo "Extracting embeddings with:"
echo "  - Window size: 3s"
echo "  - Hop size: 3s (non-overlapping)"
echo "  - Output dimension: 6522"
echo ""

# Extract Training Set
echo "[1/2] Extracting Training_Set embeddings..."
python extract_embeddings.py \
  --wav-dir /data/msc-proj/Training_Set \
  --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
  --save-in-place

echo ""
echo "[2/2] Extracting Validation_Set_DSAI_2025_2026 embeddings..."
python extract_embeddings.py \
  --wav-dir /data/msc-proj/Validation_Set_DSAI_2025_2026 \
  --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
  --save-in-place

echo ""
echo "=========================================="
echo "Extraction Complete!"
echo "=========================================="
echo "All embeddings saved as *_BDnet.npy files alongside audio files"
echo "Each file contains shape: (num_windows, 6522)"
echo ""
echo "Next steps:"
echo "  1. Run training: python train.py --config-name train_birdnet +exp_name='BirdNet'"
echo "=========================================="
