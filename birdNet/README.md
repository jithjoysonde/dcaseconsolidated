This readme is clone from . My notes are in !- HERE-!


!-
changed files from original

1.copywavefile.txt

2.data_prep py

3./config/train.yaml

4. Read me :)

5. BirdNet pipeline. see extraction guide https://github.com/jithjoysonde/dcase-2024t5/blob/birdnet_experiments/BACKGROUND_EXTRACTION_GUIDE.md
implementation details https://github.com/jithjoysonde/dcase-2024t5/blob/birdnet_experiments/BIRDNET_TEMPORAL_IMPLEMENTATION.md 
as well BirdNet downloads required (see below) before proceeding with the BirdNet Pipeline
-!
# Baseline for DCASE 2024 Task 5

The automatic detection and analysis of bioacoustic sound events is a valuable tool for assessing animal populations and their behaviours, facilitating conservation efforts, and providing insights into ecological dynamics. By leveraging advanced machine learning and computational bioacoustics techniques, it is possible to decode the complex acoustic signals of diverse species.

**Few-shot learning is a highly promising paradigm for sound event detection.** It describes tasks in which an algorithm must make predictions given only a few instances of each class, contrary to standard supervised learning paradigm. The main objective is to find reliable algorithms that are capable of dealing with data sparsity, class imbalance and noisy/busy environments.

**It is also an extremely good fit to the needs of users in bioacoustics, in which increasingly large acoustic datasets commonly need to be labelled for events of an identified category** (e.g. species or call-type), even though this category might not be known in other datasets or have any yet-known label. While satisfying user needs, this will also benchmark few-shot learning for the wider domain of sound event detection (SED).

## Description

This repo is built upon the [DCASE 2022 submission from Surrey](https://github.com/haoheliu/DCASE_2022_Task_5) for the [DCASE 2024 Challenge task 5 few-shot bio-acoustic detection](https://dcase.community/challenge2023/task-few-shot-bioacoustic-event-detection).


|    Method    | Precision (%) | Recall (%) | F-measure (%) |
| :----------: | :-----------: | :--------: | :-----------: |
| **Proposed** |     56.18     |   48.64   |     52.14     |

The species-wise perfomrnace of this system:

![species-wise performance](assets/species-wise_performance.png)

## How to train from scratch

1. Prepare environment

```bash
# clone project
git clone https://github.com/c4dm/dcase-few-shot-bioacoustic.git
cd baselines/dcase2024_task5

# create virtual environment
# we use python == 3.7 (requirements file has been updated to 3.11 thanks to Yue Yang)
python -m venv ./dcase_t5
source ./dcase_t5/bin/activate
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Prepare training and validation data. Dataset will be automatically downloaded.

```bash
python3 data_preparation.py --data-dir /dir/to/local/dataset
```

3. **Download BirdNET Model (Required for BirdNET-based experiments)**

The BirdNET pre-trained model needs to be downloaded and placed in the correct directory for feature extraction.

**Model Required:** `BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`

**Download Location:** Download the BirdNET V2.4 model from the [BirdNET-Analyzer GitHub repository](https://github.com/kahst/BirdNET-Analyzer) or from the [TFLite model releases](https://github.com/kahst/BirdNET-Analyzer/tree/main/checkpoints/V2.4).

**Expected Directory Structure:**
```
birdnet/
└── weights/
    └── model/
        └── V2.4/
            ├── BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite  (download this file)
            └── BirdNET_GLOBAL_6K_V2.4_Labels.txt         (already included)
```

**Installation:**
```bash
# Create directory if it doesn't exist
mkdir -p birdnet/weights/model/V2.4

# Download the model file (adjust URL if needed)
# Option 1: Using wget
wget -O birdnet/weights/model/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite \
  https://github.com/kahst/BirdNET-Analyzer/raw/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite

# Option 2: Download manually and place in the directory
# Download from: https://github.com/kahst/BirdNET-Analyzer/tree/main/checkpoints/V2.4
# Place the .tflite file in: birdnet/weights/model/V2.4/
```

**Note:** This model is only required if you plan to use BirdNET-based feature extraction or training configurations (e.g., `train_birdnet.yaml`).

4. Change the root directory in [train.yaml](config/train.yaml).

```yaml
path:
  root_dir: <your-root-directory>
```

5. Train model with default configuration. Find more details in [train.yaml](config/train.yaml).
6. Make your own training recipe. Ones can refer to [train.yaml](config/train.yaml) for more training options and [run.sh](run.sh) for more examples.

```bash
# use PCEN as audio features and disable negative hard sampling
python3 train.py features.feature_types="pcen" train_param.negative_train_contrast=false
```

7. Train model with CPU only.

```bash
# train on CPU
python train.py trainer.gpus=0
```

## Cite this work

```bibtex
@misc{liang2024mind,
      title={Mind the Domain Gap: a Systematic Analysis on Bioacoustic Sound Event Detection}, 
      author={Jinhua Liang and Ines Nolasco and Burooj Ghani and Huy Phan and Emmanouil Benetos and Dan Stowell},
      year={2024},
      eprint={2403.18638},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}

@techreport{Liu2022a,
    Author = "Liu, Haohe and Liu, Xubo and Mei, Xinhao and Kong, Qiuqiang and Wang, Wenwu and Plumbley, Mark D",
    title = "SURREY SYSTEM FOR DCASE 2022 TASK 5 : FEW-SHOT BIOACOUSTIC EVENT DETECTION WITH SEGMENT-LEVEL METRIC LEARNING",
    institution = "DCASE2022 Challenge Technical Report",
    year = "2022"
}
```

## Acknowledgement

> This repository is implemented based on @haoheliu's amazing [repository](https://github.com/haoheliu/DCASE_2022_Task_5). Thank @haoheliu for his contribution to the community.
