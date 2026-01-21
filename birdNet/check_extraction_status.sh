#!/bin/bash
# Check Background Extraction Status
# Run this anytime to see current progress

TRAINING_LOG="/data/4joyson/bdnt/birdnet/extraction_logs/training/combined.log"
VALIDATION_LOG="/data/4joyson/bdnt/birdnet/extraction_logs/validation/combined.log"
STATUS_FILE="/data/4joyson/bdnt/birdnet/extraction_logs/status/extraction_status.txt"
PID_FILE="/data/4joyson/bdnt/birdnet/extraction_logs/status/extraction.pid"

echo "=========================================="
echo "BirdNet Extraction Status"
echo "=========================================="
echo ""

# Check if process is running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Process is running (PID: $PID)"
    else
        echo "⚠ Process PID $PID not found (may have completed)"
    fi
else
    echo "ℹ No process information found"
fi

echo ""
echo "========== Training Set Progress =========="
if [ -f "$TRAINING_LOG" ]; then
    # Count completed classes
    COMPLETED=$(grep -c "✓ Completed: Training_Set" "$TRAINING_LOG" || echo "0")
    TOTAL=5
    echo "Classes completed: $COMPLETED/$TOTAL"
    
    # Show last few lines
    echo ""
    echo "Latest updates:"
    tail -5 "$TRAINING_LOG"
else
    echo "Log not found"
fi

echo ""
echo "========== Validation Set Progress =========="
if [ -f "$VALIDATION_LOG" ]; then
    # Count completed classes
    COMPLETED=$(grep -c "✓ Completed: Validation_Set" "$VALIDATION_LOG" || echo "0")
    TOTAL=3
    echo "Classes completed: $COMPLETED/$TOTAL"
    
    # Show last few lines
    echo ""
    echo "Latest updates:"
    tail -5 "$VALIDATION_LOG"
else
    echo "Validation extraction not yet started"
fi

echo ""
echo "========== Overall Statistics =========="

# Count files
TRAINING_COUNT=$(find /data/msc-proj/Training_Set -name "*_BDnet.npy" 2>/dev/null | wc -l)
VALIDATION_COUNT=$(find /data/msc-proj/Validation_Set_DSAI_2025_2026 -name "*_BDnet.npy" 2>/dev/null | wc -l)
TOTAL_COUNT=$((TRAINING_COUNT + VALIDATION_COUNT))

echo "Extracted embeddings:"
echo "  Training Set: $TRAINING_COUNT files"
echo "  Validation Set: $VALIDATION_COUNT files"
echo "  Total: $TOTAL_COUNT files"

echo ""
if [ -f "$STATUS_FILE" ]; then
    cat "$STATUS_FILE"
fi

echo ""
echo "=========================================="
echo "To monitor progress in real-time:"
echo "  tail -f $TRAINING_LOG"
echo "=========================================="
