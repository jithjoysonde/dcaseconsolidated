# DCASE 2024 Task 5 - Model Comparison Repository

This repository contains four different implementations for DCASE 2024 Task 5 (Few-Shot Bioacoustic Sound Event Detection):

1. **baseline1** - Original DCASE 2024 baseline
2. **baseline2** - Modified baseline with improvements
3. **birdNet** - BirdNet-based implementation with temporal features
4. **v2v3v4Mamba** - State-space model (Mamba) implementation

---

## Directory Structure

```
/data/4joyson/consdtd/
├── baseline1/              # Original baseline implementation
│   └── baselines/
│       └── dcase2024_task5/
├── baseline2/              # Modified baseline
├── birdNet/                # BirdNet-based approach
└── v2v3v4Mamba/            # Multiple models
```

## 🐍 Virtual Environment

**For baseline1, baseline2, and birdNet:**
- Shared virtual environment: `/data/4joyson/dcase_t5`
- All dependencies are pre-installed for these three projects

**For v2v3v4Mamba:**
- Requires its own separate environment (see v2v3v4Mamba/README.md)

---

### Activate the Shared Environment

```bash
source /data/4joyson/dcase_t5/bin/activate
```

---

## Baseline 1 ( DCASE 2024 Baseline with modification)

### Directory
```bash
cd /data/4joyson/consdtd/baseline1/baselines/dcase2024_task5
```

### Available Checkpoints

**Checkpoint 1: Best F-measure**
- **Path:** `/export/home/4joyson/infhome/baseline/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/logs/experiments/runs/baseline/2025-12-21_23-02-28/checkpoints/epoch_011_val_acc_0.69.ckpt`
- **Performance:**
  - F-measure: 43.19%
  - F-measure (avg): 43.32%

**Checkpoint 2: Best F-measure (avg)**
- **Path:** `/export/home/4joyson/infhome/baseline/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/logs/experiments/runs/baseline/2025-12-21_22-21-20/checkpoints/epoch_005_val_acc_0.68.ckpt`
- **Performance:**
  - F-measure: 42.47%
  - F-measure (avg): 43.70%

### Training

# Make sure you're in the correct directory
cd /data/4joyson/consdtd/baseline1/baselines/dcase2024_task5

# Activate virtual environment
source /data/4joyson/dcase_t5/bin/activate

# Data prep
python3 data_preparation.py --data-dir /data/msc-proj

# Train



# Train from scratch
python train.py
python3 train.py features.feature_types="pcen"  train_param.negative_train_contrast=false
```

### Testing/Evaluation

```bash
# Test with checkpoint 1 (best F-measure)
python test.py ckpt_path=/export/home/4joyson/infhome/baseline/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/logs/experiments/runs/baseline/2025-12-21_23-02-28/checkpoints/epoch_011_val_acc_0.69.ckpt

# Test with checkpoint 2 (best F-measure avg)
python test.py ckpt_path=/export/home/4joyson/infhome/baseline/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/logs/experiments/runs/baseline/2025-12-21_22-21-20/checkpoints/epoch_005_val_acc_0.68.ckpt
```

### Configuration
- Edit configs as needed in `configs/` directory
- Main training config: `configs/train.yaml`
- Main testing config: `configs/test.yaml`

---

## Baseline 2 (Modified Baseline with enahcements for tesing and feature loading)

### Directory
```bash
cd /data/4joyson/consdtd/baseline2
```

### Available Checkpoints

**Checkpoint 1: Best Performance**
- **Path:** `/data/4joyson/proj/logs/experiments/runs/baseline/2026-01-17_07-26-01/checkpoints/epoch_002_val_acc_0.80.ckpt`
- **Performance:**
  - F-measure: 48.15%
  - F-measure (avg): 53.86%

**Checkpoint 2: Alternative**
- **Path:** `/data/4joyson/proj/logs/experiments/runs/baseline/2026-01-17_04-17-25/checkpoints/epoch_004_val_acc_0.69.ckpt`
- **Performance:**
  - F-measure: 48.39%
  - F-measure (avg): 50.36%

### Training

```bash
# Make sure you're in the correct directory
cd /data/4joyson/consdtd/baseline2

# Activate virtual environment
source /data/4joyson/dcase_t5/bin/activate

# Train from scratch
python3 train.py features.feature_types="pcen@mfcc"  train_param.negative_train_contrast=false exp_name="jjpcenmfccNTCF"
python3 train.py features.feature_types="logmel" train_param.negative_train_contrast=false exp_name="jjpcenNTCF" 
# Or use the run script for multiple feature training
./run.sh
```

### Testing/Evaluation

```bash
# Test with checkpoint 1 (best overall)
python test.py ckpt_path=/data/4joyson/proj/logs/experiments/runs/baseline/2026-01-17_07-26-01/checkpoints/epoch_002_val_acc_0.80.ckpt

# Test with checkpoint 2
python test.py ckpt_path=/data/4joyson/proj/logs/experiments/runs/baseline/2026-01-17_04-17-25/checkpoints/epoch_004_val_acc_0.69.ckpt
```

### Configuration
- Edit configs as needed in `configs/` directory
- Main training config: `configs/train.yaml`
- Main testing config: `configs/test.yaml`

---



## v2v3v4Mamba (State-Space Model Implementation)

### Directory
```bash
cd /data/4joyson/consdtd/v2v3v4Mamba
```

### Important Notes

⚠️ **v2v3v4Mamba uses a DIFFERENT virtual environment!**

This implementation:
- Uses Mamba state-space models (requires `mamba-ssm>=1.0.1`)
- Requires CUDA-compatible installation
- Has its own comprehensive README

### Setup & Usage

Please refer to the dedicated README:
```bash
cd /data/4joyson/consdtd/v2v3v4Mamba
cat README.md
```

Key differences from other baselines:
- Uses PyTorch Lightning 2.0+ (vs 1.6.0)
- Requires `mamba-ssm`, `causal-conv1d`, `einops`
- Uses flexible version requirements (`>=` instead of `==`)
- Includes code quality tools (ruff, black, isort)

---

## 📊 Performance Comparison

| Model | F-measure (%) | F-measure avg (%) | Notes |
|-------|---------------|-------------------|-------|
| **Baseline 1 (ckpt1)** | 43.19 | 43.32 | epoch_011 |
| **Baseline 1 (ckpt2)** | 42.47 | 43.70 | epoch_005 |
| **Baseline 2 (ckpt1)** | 48.15 | 53.86 | epoch_002 ⭐ Best avg |
| **Baseline 2 (ckpt2)** | 48.39 | 50.36 | epoch_004 |
| **BirdNet** | TBD | TBD | Requires feature extraction |
| **v2v3v4Mamba** | TBD | TBD | See v2v3v4Mamba/README.md |

---

## 🔧 Common Configuration

### Data Paths

Update data paths in the respective config files:
- Training data: Set in `configs/datamodule/*.yaml`
- Evaluation data: Set in `configs/test.yaml`
- Test data: Set in `configs/test.yaml`

### Typical Config Structure

```yaml
datamodule:
  train_dir: /path/to/training/data
  eval_dir: /path/to/validation/data
  test_dir: /path/to/test/data
```

---

## 📝 Requirements Files

### Baseline1, Baseline2, BirdNet
- Similar dependencies (156-158 lines)
- Python 3.11 compatible
- PyTorch 2.1.0, PyTorch Lightning 1.6.0
- BirdNet adds: `tensorflow==2.15`

### v2v3v4Mamba
- Minimal requirements (48 lines)
- PyTorch 2.0+, PyTorch Lightning 2.0+
- Unique: Mamba SSM, causal-conv1d, einops
- Includes audio evaluation metrics: pesq, pystoi, sed_eval

---


## BirdNet (BirdNet-based Implementation)

### Directory
```bash
cd /data/4joyson/consdtd/birdNet
```

### Prerequisites

**IMPORTANT:** BirdNet requires downloading pretrained models before use.

Follow the instructions in:
- [BirdNet Extraction Guide](https://github.com/jithjoysonde/dcase-2024t5/blob/birdnet_experiments/BACKGROUND_EXTRACTION_GUIDE.md)
- [BirdNet Implementation Details](https://github.com/jithjoysonde/dcase-2024t5/blob/birdnet_experiments/BIRDNET_TEMPORAL_IMPLEMENTATION.md)

### Feature Extraction (Required First Step)

BirdNet uses a two-stage process:
1. Extract BirdNet features
2. Train/test on extracted features

```bash
# Activate virtual environment
source /data/4joyson/dcase_t5/bin/activate

# Extract features - Simple parallel version
./extract_birdnet_parallel_simple.sh

# OR extract features - Full parallel version
./extract_birdnet_parallel.sh

# OR extract features with background extraction
./extract_birdnet_background.sh

# Check extraction status
./check_extraction_status.sh
```

### Training

```bash
# Make sure features are extracted first!
cd /data/4joyson/consdtd/birdNet

# Activate virtual environment
source /data/4joyson/dcase_t5/bin/activate

# Train the model
python train.py
```

### Testing/Evaluation

```bash
# Test the BirdNet model
python test_birdnet_model.py

# OR use the debug test script
python debug_birdnet_test.py
```

### Configuration
- Main config: `configs/train.yaml`
- BirdNet requires TensorFlow 2.15 (included in requirements)
- Modified data preparation: `data_preparation.py`

### Helper Scripts
- `setup_gpu.sh` - GPU setup utilities
- `start_background_extraction.sh` - Background feature extraction
- `verify_encoder_gradients.py` - Verify model gradients