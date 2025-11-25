#
# Model factory for MolecularDiT
#

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from functools import partial
from model.layers import (
    TimestepEmbedder,
    FourierPositionEmbedder,
    EfficientSelfAttentionLayer,
    DiTBlock,
    HomogenTrunk,
)
from model.molecular_dit import MolecularDiT


def create_molecular_dit(
    hidden_size=512,
    num_heads=8,
    depth=12,
    mlp_ratio=4.0,
    max_atoms=80,
    num_atom_types=13,
    dreams_embedding_dim=1024,
    pos_embed_dim=258,
    atom_n_queries=None,  # None = full attention
    atom_n_keys=None,
    dreams_dropout_prob=0.1,
    use_length_condition=True,
    use_swiglu=True,
):
    """
    Create a MolecularDiT model.
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        depth: Number of transformer layers
        mlp_ratio: MLP expansion ratio
        max_atoms: Maximum number of atoms
        num_atom_types: Number of element types
        dreams_embedding_dim: DreaMS embedding dimension
        pos_embed_dim: Fourier position embedding dimension (must be divisible by 6)
        atom_n_queries: Query window for local attention (None = full)
        atom_n_keys: Key/value window for local attention
        dreams_dropout_prob: Dropout for classifier-free guidance
        use_length_condition: Whether to condition on number of atoms
        use_swiglu: Use SwiGLU in MLP
    """
    
    # Time embedder
    time_embedder = TimestepEmbedder(hidden_size=hidden_size)
    
    # Position embedder (Fourier features)
    pos_embedder = FourierPositionEmbedder(embed_dim=pos_embed_dim)
    
    # Attention layer factory
    attn_layer_fn = partial(
        EfficientSelfAttentionLayer,
        hidden_size=hidden_size,
        num_heads=num_heads,
        qkv_bias=True,
        qk_norm=True,
    )
    
    # DiT block factory
    block_fn = partial(
        DiTBlock,
        self_attention_layer=attn_layer_fn,
        hidden_size=hidden_size,
        mlp_ratio=mlp_ratio,
        use_swiglu=use_swiglu,
    )
    
    # Atom transformer (stack of DiT blocks)
    atom_transformer = HomogenTrunk(block=block_fn, depth=depth)
    
    # Create model
    model = MolecularDiT(
        atom_transformer=atom_transformer,
        time_embedder=time_embedder,
        pos_embedder=pos_embedder,
        hidden_size=hidden_size,
        num_heads=num_heads,
        output_channels=3,
        atom_n_queries=atom_n_queries,
        atom_n_keys=atom_n_keys,
        dreams_embedding_dim=dreams_embedding_dim,
        num_atom_types=num_atom_types,
        dreams_dropout_prob=dreams_dropout_prob,
        use_length_condition=use_length_condition,
    )
    
    return model


if __name__ == "__main__":
    print("Testing model factory...")
    print("=" * 80)
    
    import torch
    
    # Create small model for testing
    model = create_molecular_dit(
        hidden_size=256,
        num_heads=4,
        depth=4,
        max_atoms=80,
    )
    
    print(f"Model created successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    max_atoms = 80
    
    feats = {
        'ref_pos': torch.randn(batch_size, max_atoms, 3),
        'ref_element': torch.randn(batch_size, max_atoms, 13),
        'atom_pad_mask': torch.ones(batch_size, max_atoms).bool(),
        'dreams_embedding': torch.randn(batch_size, 1024),
        'num_atoms': torch.tensor([45, 62]),
    }
    
    noised_pos = torch.randn(batch_size, max_atoms, 3)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        output = model(noised_pos, t, feats)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape: {noised_pos.shape}")
    print(f"  Output shape: {output['predict_velocity'].shape}")
    
    print("\n" + "=" * 80)
    print("✓ Model factory tests passed!")
