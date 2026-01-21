"""
BirdNet Encoder Wrapper for Prototypical Networks.
This is a pass-through encoder that uses pre-computed BirdNet embeddings.
"""
import torch
import torch.nn as nn


class BirdNetEncoder(nn.Module):
    """
    Pass-through encoder for pre-computed BirdNet embeddings.
    Simply applies temporal pooling to get fixed-size representations.
    """
    
    def __init__(self, embedding_dim=6522, pooling='mean'):
        """
        Args:
            embedding_dim: Dimension of BirdNet embeddings (default: 6522)
            pooling: Type of temporal pooling ('mean', 'max', or 'both')
        """
        super(BirdNetEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        
        if pooling == 'both':
            # Concatenate mean and max pooling
            self.output_dim = embedding_dim * 2
        else:
            self.output_dim = embedding_dim
    
    def forward(self, x):
        """
        Forward pass through BirdNet encoder.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, 6522)
               These are pre-computed BirdNet embeddings loaded by the dataloader
        
        Returns:
            output: Tensor of shape (batch_size, output_dim)
        """
        # x shape: (batch, time_steps, 6522)
        # No need to detach - embeddings are just input features
        
        if self.pooling == 'mean':
            # Mean pooling over time dimension
            output = torch.mean(x, dim=1)  # (batch, 1280)
        elif self.pooling == 'max':
            # Max pooling over time dimension
            output = torch.max(x, dim=1)[0]  # (batch, 1280)
        elif self.pooling == 'both':
            # Concatenate mean and max pooling
            mean_pool = torch.mean(x, dim=1)  # (batch, 1280)
            max_pool = torch.max(x, dim=1)[0]  # (batch, 1280)
            output = torch.cat([mean_pool, max_pool], dim=1)  # (batch, 2560)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        
        return output


class BirdNetEncoderWithProjection(nn.Module):
    """
    BirdNet encoder with optional projection layer for dimensionality reduction.
    """
    
    def __init__(self, embedding_dim=1280, output_dim=None, pooling='mean', dropout=0.1):
        """
        Args:
            embedding_dim: Dimension of BirdNet embeddings (default: 1280)
            output_dim: If specified, project to this dimension (default: None, no projection)
            pooling: Type of temporal pooling ('mean', 'max', or 'both')
            dropout: Dropout rate for projection layer
        """
        super(BirdNetEncoderWithProjection, self).__init__()
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        
        if pooling == 'both':
            pooled_dim = embedding_dim * 2
        else:
            pooled_dim = embedding_dim
        
        self.output_dim = output_dim if output_dim else pooled_dim
        
        # Optional projection layer
        if output_dim:
            self.projection = nn.Sequential(
                nn.Linear(pooled_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.projection = None
    
    def forward(self, x):
        """
        Forward pass with optional projection.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, embedding_dim)
        
        Returns:
            output: Tensor of shape (batch_size, output_dim)
        """
        # Temporal pooling
        if self.pooling == 'mean':
            pooled = torch.mean(x, dim=1)
        elif self.pooling == 'max':
            pooled = torch.max(x, dim=1)[0]
        elif self.pooling == 'both':
            mean_pool = torch.mean(x, dim=1)
            max_pool = torch.max(x, dim=1)[0]
            pooled = torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        
        # Optional projection
        if self.projection:
            output = self.projection(pooled)
        else:
            output = pooled
        
        return output


if __name__ == "__main__":
    # Test the encoder
    batch_size = 4
    time_steps = 17
    embedding_dim = 1280
    
    # Create dummy input
    x = torch.randn(batch_size, time_steps, embedding_dim)
    
    # Test basic encoder
    print("Testing BirdNetEncoder...")
    encoder = BirdNetEncoder(embedding_dim=1280, pooling='mean')
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {embedding_dim})")
    
    # Test encoder with projection
    print("\nTesting BirdNetEncoderWithProjection...")
    encoder_proj = BirdNetEncoderWithProjection(
        embedding_dim=1280, 
        output_dim=512, 
        pooling='both'
    )
    output_proj = encoder_proj(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_proj.shape}")
    print(f"Expected: ({batch_size}, 512)")
