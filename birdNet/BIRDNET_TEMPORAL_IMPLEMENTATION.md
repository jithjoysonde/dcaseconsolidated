# BirdNet Temporal Embeddings Integration - Implementation Summary

## Overview
We've implemented Extraction with Sliding Window to convert BirdNet's single-vector embeddings into temporal sequences suitable for few-shot learning.




## Performance Characteristics

| Metric | Value |
|--------|-------|
| Window Size | 3.0 seconds |
| Hop Size | 3.0 seconds (non-overlapping) |
| Embedding Dimension | 6522 |
| Temporal Resolution | ~0.33 Hz (one window per 3 seconds) |
| Model | BirdNet V2.4 TFLite |



## Extraction Script



```bash
cd /data/4joyson/bdnt
chmod +x extract_birdnet_sliding_window.sh
./extract_birdnet_sliding_window.sh
```

Or manually:

```bash
cd /data/4joyson/bdnt/birdnet
source /data/4joyson/dcase_t5/bin/activate

# Extract Training Set
python extract_embeddings.py \
  --wav-dir /data/msc-proj/Training_Set \
  --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
  --save-in-place

# Extract Validation Set
python extract_embeddings.py \
  --wav-dir /data/msc-proj/Validation_Set_DSAI_2025_2026 \
  --model weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
  --save-in-place
```

## Output Format

Each audio file gets a corresponding `_BDnet.npy` file:



## Training After Extraction

Once extraction completes:

```bash
cd /data/4joyson/bdnt
python train.py --config-name train_birdnet +exp_name="BirdNet"
```

**What happens during training**:
1. Dataloader loads `(num_windows, 6522)` embeddings
2. BirdNetEncoder applies temporal pooling: `(num_windows, 6522)` → `(1, 6522)`
3. Prototypical classifier processes 6522-dim vectors
4. PSDS evaluation unchanged (uses same metrics pipeline)

## Architecture Diagram

┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                      │
├─────────────────────────────────────────────────────────────┤

│  Audio Files → BirdNet Extraction (3s windows)             │
│       ↓                                                      │
│  Temporal Embeddings (num_windows, 6522)                   │
│       ↓                                                      │
│  BirdNetDataset → Segment Selection → Random Crop          │
│       ↓                                                      │
│  Batch: (batch_size, seg_len, 6522)                        │
│       ↓                                                      │
│  BirdNetEncoder (Temporal Pooling) → (batch_size, 6522)    │
│       ↓                                                      │
│  Prototypical Classifier → 5-way Classification            │
│       ↓                                                      │
│  Contrastive Loss + PSDS Evaluation                         │
│                                                             │

