#!/bin/bash
# Start Background Extraction with nohup
# This script starts the extraction in the background so you can safely logout

cd /data/4joyson/bdnt

# Make extraction script executable
chmod +x extract_birdnet_background.sh

echo "=========================================="
echo "Starting BirdNet Background Extraction"
echo "=========================================="
echo ""

# Start in background with nohup (survives logout)
nohup bash -c 'cd /data/4joyson/bdnt && ./extract_birdnet_background.sh' \
    > /dev/null 2>&1 &

MAIN_PID=$!

echo "✓ Background process started!"
echo "  Main PID: $MAIN_PID"
echo ""
echo "Log Files:"
echo "  Training Set: /data/4joyson/bdnt/birdnet/extraction_logs/training/combined.log"
echo "  Validation Set: /data/4joyson/bdnt/birdnet/extraction_logs/validation/combined.log"
echo "  Status: /data/4joyson/bdnt/birdnet/extraction_logs/status/extraction_status.txt"
echo ""
echo "Commands to monitor progress:"
echo "  # Watch Training Set extraction"
echo "  tail -f /data/4joyson/bdnt/birdnet/extraction_logs/training/combined.log"
echo ""
echo "  # Watch Validation Set extraction"
echo "  tail -f /data/4joyson/bdnt/birdnet/extraction_logs/validation/combined.log"
echo ""
echo "  # Watch specific class (e.g., HT)"
echo "  tail -f /data/4joyson/bdnt/birdnet/extraction_logs/training/HT.log"
echo ""
echo "  # Check overall status"
echo "  cat /data/4joyson/bdnt/birdnet/extraction_logs/status/extraction_status.txt"
echo ""
echo "  # Count completed embeddings"
echo "  find /data/msc-proj -name '*_BDnet.npy' | wc -l"
echo ""
echo "Estimated time: 6-18 hours (depends on system load)"
echo ""
echo "You can now safely logout. The extraction will continue in background."
echo "=========================================="
