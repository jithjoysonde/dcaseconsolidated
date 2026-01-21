#!/bin/bash
# BirdNet Feature Extraction and Prediction Pipeline
# Usage: bash run_birdnet_pipeline.sh

set -e  # Exit on error

# Configuration
MODEL_PATH="birdnet/weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
TRAIN_DATA="/data/msc-proj/Training_Set"
VAL_DATA="/data/msc-proj/Validation_Set_DSAI_2025_2026"
TEST_DATA="/data/msc-proj/Evaluation_Set_DSAI_2025_2026"

TRAIN_CSV="/data/msc-proj/Training_Set_classes.csv"
VAL_CSV="/data/msc-proj/Validation_Set_classes_DSAI_2025_2026.csv"

FEATURES_DIR="birdnet/features"
OUTPUT_CSV="birdnet/predictions.csv"

# Set PYTHONPATH
export PYTHONPATH="/data/4joyson/bdnt:$PYTHONPATH"

echo "========================================"
echo "BirdNet Feature Extraction & Prediction"
echo "========================================"

# Step 1: Extract features from training data
echo ""
echo "Step 1/4: Extracting features from training data..."
python birdnet/extract_embeddings.py \
  --wav-dir "$TRAIN_DATA" \
  --model "$MODEL_PATH" \
  --out-dir "$FEATURES_DIR/train"

# Step 2: Extract features from validation data
echo ""
echo "Step 2/4: Extracting features from validation data..."
python birdnet/extract_embeddings.py \
  --wav-dir "$VAL_DATA" \
  --model "$MODEL_PATH" \
  --out-dir "$FEATURES_DIR/val"

# Step 3: Extract features from test data
echo ""
echo "Step 3/4: Extracting features from test/evaluation data..."
python birdnet/extract_embeddings.py \
  --wav-dir "$TEST_DATA" \
  --model "$MODEL_PATH" \
  --out-dir "$FEATURES_DIR/test"

# Step 4: Run predictions
echo ""
echo "Step 4/4: Running predictions..."
python birdnet/run_prediction.py \
  --train-dir "$FEATURES_DIR/train" \
  --val-dir "$FEATURES_DIR/val" \
  --test-dir "$FEATURES_DIR/test" \
  --test-wav-dir "$TEST_DATA" \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --output-csv "$OUTPUT_CSV" \
  --few-shot

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo "Predictions saved to: $OUTPUT_CSV"
echo ""
echo "Feature counts:"
echo "  Training: $(ls $FEATURES_DIR/train/*.npy 2>/dev/null | wc -l) files"
echo "  Validation: $(ls $FEATURES_DIR/val/*.npy 2>/dev/null | wc -l) files"
echo "  Test: $(ls $FEATURES_DIR/test/*.npy 2>/dev/null | wc -l) files"
