"""Verify that gradients flow through the BirdNetEncoder."""
import torch
from src.models.components.birdnet_encoder import BirdNetEncoder

# Create encoder
encoder = BirdNetEncoder(embedding_dim=6522, pooling='mean')

# Create dummy input
batch_size = 4
time_steps = 10
x = torch.randn(batch_size, time_steps, 6522, requires_grad=True)

# Forward pass
output = encoder(x)

# Compute loss and backward
loss = output.sum()
loss.backward()

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Input requires_grad: {x.requires_grad}")
print(f"Output requires_grad: {output.requires_grad}")
print(f"Input has grad: {x.grad is not None}")
print(f"Gradient sum: {x.grad.sum().item() if x.grad is not None else 'None'}")

if x.grad is not None and x.grad.sum().item() != 0:
    print("\n✓ Gradients flow correctly through the encoder!")
else:
    print("\n✗ ERROR: Gradients are NOT flowing!")
