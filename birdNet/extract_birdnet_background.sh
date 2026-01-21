#!/bin/bash
# Background BirdNet Extraction Script with Separate Logs
# Runs in background using nohup so you can safely logout
# All processes continue running even after logout

set -e

source /data/4joyson/dcase_t5/bin/activate
cd /data/4joyson/bdnt/birdnet

# Create log directories
mkdir -p extraction_logs/training
mkdir -p extraction_logs/validation
mkdir -p extraction_logs/status

LOG_TRAINING="extraction_logs/training/combined.log"
LOG_VALIDATION="extraction_logs/validation/combined.log"
STATUS_FILE="extraction_logs/status/extraction_status.txt"
PID_FILE="extraction_logs/status/extraction.pid"

echo "=========================================="
echo "BirdNet Extraction - Background Mode"
echo "=========================================="
echo "Process started at: $(date)"
echo "PID: $$"
echo "Log files:"
echo "  Training Set: $LOG_TRAINING"
echo "  Validation Set: $LOG_VALIDATION"
echo "  Status: $STATUS_FILE"
echo ""
echo "To monitor: tail -f $LOG_TRAINING"
echo "To monitor: tail -f $LOG_VALIDATION"
echo ""

# Save PID
echo $$ > "$PID_FILE"

# ============================================
# PHASE 1: Training Set Extraction (5 classes)
# ============================================

{
    echo "=========================================="
    echo "PHASE 1: Training Set Extraction"
    echo "Started: $(date)"
    echo "=========================================="
    echo ""
    
    # Track which classes are done
    CLASSES=("HT" "JD" "BV" "WMW" "MT")
    PIDS=()
    
    echo "Starting 5 parallel extractions for Training_Set..."
    
    for CLASS in "${CLASSES[@]}"; do
        {
            echo "[$(date +%H:%M:%S)] Starting: Training_Set/$CLASS"
            python extract_embeddings.py \
                --wav-dir "/data/msc-proj/Training_Set/$CLASS" \
                --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
                --save-in-place 2>&1
            echo "[$(date +%H:%M:%S)] ✓ Completed: Training_Set/$CLASS"
        } > "extraction_logs/training/${CLASS}.log" 2>&1 &
        PIDS+=($!)
    done
    
    echo "Background processes started:"
    for i in "${!CLASSES[@]}"; do
        echo "  ${CLASSES[$i]} (PID: ${PIDS[$i]})"
    done
    echo ""
    
    # Wait for all to complete
    echo "Waiting for all Training_Set extractions to complete..."
    echo "Check progress: tail -f extraction_logs/training/*.log"
    echo ""
    
    for i in "${!PIDS[@]}"; do
        wait ${PIDS[$i]}
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] ✓ ${CLASSES[$i]} extraction complete"
        else
            echo "[$(date +%H:%M:%S)] ✗ ${CLASSES[$i]} extraction failed (exit code: $STATUS)"
        fi
    done
    
    echo ""
    echo "Training Set Phase Complete!"
    echo "Completed: $(date)"
    echo ""
    
} >> "$LOG_TRAINING" 2>&1

# ============================================
# PHASE 2: Validation Set Extraction (3 classes)
# ============================================

{
    echo "=========================================="
    echo "PHASE 2: Validation Set Extraction"
    echo "Started: $(date)"
    echo "=========================================="
    echo ""
    
    CLASSES=("ME" "PB" "PB24")
    PIDS=()
    
    echo "Starting 3 parallel extractions for Validation_Set_DSAI_2025_2026..."
    
    for CLASS in "${CLASSES[@]}"; do
        {
            echo "[$(date +%H:%M:%S)] Starting: Validation_Set_DSAI_2025_2026/$CLASS"
            python extract_embeddings.py \
                --wav-dir "/data/msc-proj/Validation_Set_DSAI_2025_2026/$CLASS" \
                --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
                --save-in-place 2>&1
            echo "[$(date +%H:%M:%S)] ✓ Completed: Validation_Set_DSAI_2025_2026/$CLASS"
        } > "extraction_logs/validation/${CLASS}.log" 2>&1 &
        PIDS+=($!)
    done
    
    echo "Background processes started:"
    for i in "${!CLASSES[@]}"; do
        echo "  ${CLASSES[$i]} (PID: ${PIDS[$i]})"
    done
    echo ""
    
    # Wait for all to complete
    echo "Waiting for all Validation_Set extractions to complete..."
    echo "Check progress: tail -f extraction_logs/validation/*.log"
    echo ""
    
    for i in "${!PIDS[@]}"; do
        wait ${PIDS[$i]}
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] ✓ ${CLASSES[$i]} extraction complete"
        else
            echo "[$(date +%H:%M:%S)] ✗ ${CLASSES[$i]} extraction failed (exit code: $STATUS)"
        fi
    done
    
    echo ""
    echo "Validation Set Phase Complete!"
    echo "Completed: $(date)"
    echo ""
    
} >> "$LOG_VALIDATION" 2>&1

# ============================================
# COMPLETION
# ============================================

{
    echo "=========================================="
    echo "✓ ALL EXTRACTIONS COMPLETE!"
    echo "=========================================="
    echo "Completion time: $(date)"
    echo ""
    
    # Count extracted files
    TRAINING_COUNT=$(find /data/msc-proj/Training_Set -name "*_BDnet.npy" | wc -l)
    VALIDATION_COUNT=$(find /data/msc-proj/Validation_Set_DSAI_2025_2026 -name "*_BDnet.npy" | wc -l)
    
    echo "Summary:"
    echo "  Training Set embeddings: $TRAINING_COUNT files"
    echo "  Validation Set embeddings: $VALIDATION_COUNT files"
    echo "  Total: $((TRAINING_COUNT + VALIDATION_COUNT)) files"
    echo ""
    
    echo "Log locations:"
    echo "  - $LOG_TRAINING"
    echo "  - $LOG_VALIDATION"
    echo ""
    
    echo "Next step: Start training"
    echo "  cd /data/4joyson/bdnt"
    echo "  python train.py --config-name train_birdnet +exp_name='BirdNet'"
    echo ""
    echo "=========================================="
    
} | tee -a "$LOG_TRAINING" "$LOG_VALIDATION"

# Update status file
echo "COMPLETE" > "$STATUS_FILE"
echo "Completed at: $(date)" >> "$STATUS_FILE"
