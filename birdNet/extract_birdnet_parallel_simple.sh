#!/bin/bash
# Parallel BirdNet Extraction - No dependencies version
# Runs 5 extraction processes simultaneously using background jobs

set -e

source /data/4joyson/dcase_t5/bin/activate
cd /data/4joyson/bdnt/birdnet

echo "=========================================="
echo "GPU-Optimized BirdNet Extraction"
echo "=========================================="
echo "Multi-threaded CPU inference (8 threads/process)"
echo "Running 5 parallel processes"
echo ""

# Create log directory
mkdir -p extraction_logs

echo "Phase 1: Training Set (5 classes in parallel)"
echo "=========================================="

# Start all 5 Training Set extractions in background
python extract_embeddings.py \
    --wav-dir /data/msc-proj/Training_Set/HT \
    --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --save-in-place \
    > extraction_logs/HT.log 2>&1 &
PID_HT=$!

python extract_embeddings.py \
    --wav-dir /data/msc-proj/Training_Set/JD \
    --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --save-in-place \
    > extraction_logs/JD.log 2>&1 &
PID_JD=$!

python extract_embeddings.py \
    --wav-dir /data/msc-proj/Training_Set/BV \
    --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --save-in-place \
    > extraction_logs/BV.log 2>&1 &
PID_BV=$!

python extract_embeddings.py \
    --wav-dir /data/msc-proj/Training_Set/WMW \
    --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --save-in-place \
    > extraction_logs/WMW.log 2>&1 &
PID_WMW=$!

python extract_embeddings.py \
    --wav-dir /data/msc-proj/Training_Set/MT \
    --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --save-in-place \
    > extraction_logs/MT.log 2>&1 &
PID_MT=$!

echo "Started 5 extraction processes:"
echo "  HT  (PID: $PID_HT)  -> extraction_logs/HT.log"
echo "  JD  (PID: $PID_JD)  -> extraction_logs/JD.log"
echo "  BV  (PID: $PID_BV)  -> extraction_logs/BV.log"
echo "  WMW (PID: $PID_WMW) -> extraction_logs/WMW.log"
echo "  MT  (PID: $PID_MT)  -> extraction_logs/MT.log"
echo ""
echo "Monitor progress: tail -f extraction_logs/*.log"
echo ""

# Wait for all Training Set extractions to complete
echo "Waiting for Training Set to complete..."
wait $PID_HT && echo "✓ HT complete" || echo "✗ HT failed"
wait $PID_JD && echo "✓ JD complete" || echo "✗ JD failed"
wait $PID_BV && echo "✓ BV complete" || echo "✗ BV failed"
wait $PID_WMW && echo "✓ WMW complete" || echo "✗ WMW failed"
wait $PID_MT && echo "✓ MT complete" || echo "✗ MT failed"

echo ""
echo "Phase 2: Validation Set (3 classes in parallel)"
echo "=========================================="

# Start Validation Set extractions
python extract_embeddings.py \
    --wav-dir /data/msc-proj/Validation_Set_DSAI_2025_2026/ME \
    --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --save-in-place \
    > extraction_logs/ME.log 2>&1 &
PID_ME=$!

python extract_embeddings.py \
    --wav-dir /data/msc-proj/Validation_Set_DSAI_2025_2026/PB \
    --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --save-in-place \
    > extraction_logs/PB.log 2>&1 &
PID_PB=$!

python extract_embeddings.py \
    --wav-dir /data/msc-proj/Validation_Set_DSAI_2025_2026/PB24 \
    --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
    --save-in-place \
    > extraction_logs/PB24.log 2>&1 &
PID_PB24=$!

echo "Started 3 extraction processes:"
echo "  ME   (PID: $PID_ME)   -> extraction_logs/ME.log"
echo "  PB   (PID: $PID_PB)   -> extraction_logs/PB.log"
echo "  PB24 (PID: $PID_PB24) -> extraction_logs/PB24.log"
echo ""

# Wait for Validation Set
echo "Waiting for Validation Set to complete..."
wait $PID_ME && echo "✓ ME complete" || echo "✗ ME failed"
wait $PID_PB && echo "✓ PB complete" || echo "✗ PB failed"
wait $PID_PB24 && echo "✓ PB24 complete" || echo "✗ PB24 failed"

echo ""
echo "=========================================="
echo "✓ All Extractions Complete!"
echo "=========================================="
echo "Logs saved in: extraction_logs/"
echo "Embeddings: *_BDnet.npy (shape: num_windows × 6522)"
echo ""
echo "Next step:"
echo "  cd /data/4joyson/bdnt"
echo "  python train.py --config-name train_birdnet +exp_name='BirdNet'"
echo "=========================================="
