"""Debug script to check BirdNet test data and model predictions."""
import numpy as np
import torch
from src.datamodules.components.birdnet_dataset import BirdNetTestSet
from src.models.components.birdnet_encoder import BirdNetEncoder
from omegaconf import DictConfig, OmegaConf

# Create minimal config
config = OmegaConf.create({
    "path": {
        "train_dir": "/data/msc-proj/Training_Set/",
        "val_dir": "/data/msc-proj/Validation_Set_DSAI_2025_2026/",
        "test_dir": "/data/msc-proj/Validation_Set_DSAI_2025_2026/",
        "eval_dir": "/data/msc-proj/Validation_Set_DSAI_2025_2026/",
    },
    "train_param": {
        "n_shot": 5,
    },
    "eval_param": {
        "seg_len": 1.5,
        "hop_seg": 0.75,
    },
    "features": {
        "sr": 22050,
        "hop_mel": 256,
    }
})

# Create dataset
print("Creating BirdNetTestSet...")
dataset = BirdNetTestSet(
    path=config.path,
    train_param=config.train_param,
    eval_param=config.eval_param,
    features=config.features,
)

print(f"Dataset length: {len(dataset)}")

# Get first sample
print("\nGetting first sample...")
batch = dataset[0]
inner, strt_index_query, audio_path, seg_len = batch
(X_pos, X_neg, X_query, X_pos_neg, X_neg_neg, X_query_neg, 
 hop_seg, hop_seg_neg, max_len, neg_min_length) = inner

print(f"\nBatch structure:")
print(f"X_pos shape: {X_pos.shape}")
print(f"X_neg shape: {X_neg.shape}")
print(f"X_query shape: {X_query.shape}")
print(f"hop_seg: {hop_seg}")
print(f"seg_len: {seg_len}")
print(f"audio_path: {audio_path}")

# Test encoder
print("\nTesting BirdNetEncoder...")
encoder = BirdNetEncoder(embedding_dim=6522, pooling='mean')
encoder.eval()

# Test with a single sample
X_pos_tensor = torch.tensor(X_pos[:1]).float()  # Take first positive sample
print(f"Input tensor shape: {X_pos_tensor.shape}")

with torch.no_grad():
    output = encoder(X_pos_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output stats - mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

print("\nDebug completed!")
