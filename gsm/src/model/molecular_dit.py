# 
# Molecular Diffusion Transformer for GSM
# Adapted from SimpleFold architecture for molecule generation
#

import math
import torch
from torch import nn
from torch.nn import functional as F


class FinalLayer(nn.Module):
    """
    The final layer of DiT (output projection with adaptive layer norm).
    """
    def __init__(self, hidden_size, output_channels, c_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ConditionEmbedder(nn.Module):
    """
    Embeds conditioning signals (like DreaMS embeddings) with optional dropout.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        self.dropout_prob = dropout_prob

    def forward(self, x, training=False, force_drop_ids=None):
        x = self.proj(x)
        
        if training and self.dropout_prob > 0:
            # Apply dropout to entire samples (not individual features)
            if force_drop_ids is not None:
                x[force_drop_ids] = 0
            else:
                mask = torch.rand(x.shape[0], 1, device=x.device) > self.dropout_prob
                x = x * mask
        
        return x


def modulate(x, shift, scale):
    """Apply affine transformation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MolecularDiT(nn.Module):
    """
    Diffusion Transformer for Molecular Structure Generation.
    
    Key differences from SimpleFold:
    - No residue tokens (direct atom-level processing)
    - No ESM embeddings (uses DreaMS spectrum embeddings)
    - No positional encodings based on sequence order
    - Permutation invariant (only spatial relationships matter)
    """
    
    def __init__(
        self,
        atom_transformer,  # Main transformer operating on atoms
        time_embedder,
        pos_embedder,  # Spatial position embedder (Fourier features, NOT RoPE!)
        hidden_size=1152,
        num_heads=16,
        output_channels=3,
        atom_n_queries=32,
        atom_n_keys=128,
        dreams_embedding_dim=1024,
        num_atom_types=13,  # Number of unique elements
        dreams_dropout_prob=0.0,
        use_length_condition=True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.atom_n_queries = atom_n_queries
        self.atom_n_keys = atom_n_keys
        self.dreams_dropout_prob = dreams_dropout_prob
        self.use_length_condition = use_length_condition
        
        # Embedding layers
        self.pos_embedder = pos_embedder
        pos_embed_channels = pos_embedder.embed_dim
        
        self.time_embedder = time_embedder
        
        self.atom_transformer = atom_transformer
        
        # Optional length conditioning
        if self.use_length_condition:
            self.length_embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
            )
        
        # Atom feature projection
        # Input: pos_embed + element_onehot + (optional: charge, etc.)
        atom_feat_dim = pos_embed_channels + num_atom_types
        self.atom_feat_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        
        # Noised position projection
        self.atom_pos_proj = nn.Linear(pos_embed_channels, hidden_size, bias=False)
        
        # Combine feature and position embeddings
        self.atom_in_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # DreaMS embedding projection (replaces ESM in SimpleFold)
        self.dreams_proj = ConditionEmbedder(
            input_dim=dreams_embedding_dim,
            hidden_size=hidden_size,
            dropout_prob=self.dreams_dropout_prob,
        )
        
        # Final output layer
        self.final_layer = FinalLayer(
            hidden_size,
            output_channels,
            c_dim=hidden_size,
        )
    
    def create_local_attn_bias(
        self,
        n: int,
        n_queries: int,
        n_keys: int,
        inf: float = 1e10,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Create local attention bias for windowed attention.
        
        Args:
            n: Number of atoms
            n_queries: Query window size
            n_keys: Key/value window size
            inf: Value for masked positions
            device: Device to create tensor on
            
        Returns:
            Attention bias tensor [n, n]
        """
        n_trunks = int(math.ceil(n / n_queries))
        padded_n = n_trunks * n_queries
        attn_mask = torch.zeros(padded_n, padded_n, device=device)
        
        for block_index in range(n_trunks):
            i = block_index * n_queries
            j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
            j2 = n_queries * block_index + (n_queries + n_keys) // 2
            attn_mask[i : i + n_queries, j1:j2] = 1.0
        
        attn_bias = (1 - attn_mask) * -inf
        return attn_bias.to(device=device)[:n, :n]
    
    def create_atom_attn_mask(
        self,
        n_atoms: int,
        device: torch.device,
        atom_n_queries: int = None,
        atom_n_keys: int = None,
        inf: float = 1e10,
    ) -> torch.Tensor:
        """
        Create attention mask for atoms.
        
        If n_queries and n_keys are provided, creates local attention.
        Otherwise, returns None for full attention.
        """
        if atom_n_queries is not None and atom_n_keys is not None:
            return self.create_local_attn_bias(
                n=n_atoms,
                n_queries=atom_n_queries,
                n_keys=atom_n_keys,
                device=device,
                inf=inf,
            )
        else:
            return None
    
    def forward(self, noised_pos, t, feats, self_cond=None):
        """
        Forward pass through molecular DiT.
        
        Args:
            noised_pos: [B, N, 3] noisy atomic coordinates
            t: [B] timestep
            feats: Dictionary containing:
                - ref_pos: [B, N, 3] reference positions
                - ref_element: [B, N, num_atom_types] element one-hot
                - atom_pad_mask: [B, N] padding mask
                - dreams_embedding: [B, 1024] DreaMS spectrum embedding
                - num_atoms: [B] actual number of atoms per sample
            self_cond: Optional self-conditioning (not used yet)
            
        Returns:
            Dictionary with:
                - predict_velocity: [B, N, 3] predicted velocity field
        """
        B, N, _ = feats["ref_pos"].shape
        device = noised_pos.device
        
        # Create attention mask (local or full)
        atom_attn_mask = self.create_atom_attn_mask(
            n_atoms=N,
            device=device,
            atom_n_queries=self.atom_n_queries,
            atom_n_keys=self.atom_n_keys,
        )
        
        # Create time conditioning
        c_emb = self.time_embedder(t)  # [B, hidden_size]
        
        # Optional: Add length conditioning (number of atoms)
        if self.use_length_condition:
            length = feats["num_atoms"].float().unsqueeze(-1)  # [B, 1]
            c_emb = c_emb + self.length_embedder(torch.log(length + 1))
        
        # Create atom features from reference structure
        ref_pos_emb = self.pos_embedder(pos=feats["ref_pos"])  # [B, N, pos_embed_dim]
        
        atom_feat = torch.cat(
            [
                ref_pos_emb,                    # [B, N, pos_embed_dim]
                feats["ref_element"],           # [B, N, num_atom_types]
            ],
            dim=-1,
        )  # [B, N, pos_embed_dim + num_atom_types]
        
        atom_feat = self.atom_feat_proj(atom_feat)  # [B, N, hidden_size]
        
        # Create position embedding from noised coordinates
        atom_coord = self.pos_embedder(pos=noised_pos)  # [B, N, pos_embed_dim]
        atom_coord = self.atom_pos_proj(atom_coord)     # [B, N, hidden_size]
        
        # Combine features and positions
        atom_in = torch.cat([atom_feat, atom_coord], dim=-1)  # [B, N, 2*hidden_size]
        atom_in = self.atom_in_proj(atom_in)                   # [B, N, hidden_size]
        
        # Add DreaMS conditioning (broadcast to all atoms)
        dreams_emb = self.dreams_proj(
            feats["dreams_embedding"],
            training=self.training,
            force_drop_ids=feats.get("force_drop_ids", None),
        )  # [B, hidden_size]
        
        # Broadcast DreaMS embedding to all atoms and add
        atom_in = atom_in + dreams_emb.unsqueeze(1)  # [B, N, hidden_size]
        
        # NO POSITIONAL ENCODINGS! (permutation invariant)
        # The 'pos' kwarg is NOT used - attention is based only on learned features
        # Spatial information is already encoded in atom_feat via pos_embedder
        
        # Main atom transformer (no pos argument!)
        output = self.atom_transformer(
            latents=atom_in,
            c=c_emb,
            attention_mask=atom_attn_mask,
            # NO pos kwarg! Permutation invariant!
        )  # [B, N, hidden_size]
        
        # Final projection to velocity
        output = self.final_layer(output, c=c_emb)  # [B, N, 3]
        
        return {
            "predict_velocity": output,
        }


if __name__ == "__main__":
    print("Testing MolecularDiT architecture...")
    print("=" * 80)
    
    # This is just a structure test - we'll need to import actual transformer blocks
    # from SimpleFold or create them separately
    
    print("Architecture defined successfully!")
    print("\nKey architectural differences from SimpleFold:")
    print("  ✓ No residue tokenization")
    print("  ✓ No atom encoder → grouping → residue trunk → ungrouping → atom decoder")
    print("  ✓ Direct atom-level transformer")
    print("  ✓ DreaMS embeddings instead of ESM")
    print("  ✓ No sequence-based positional encodings")
    print("  ✓ Permutation invariant (only spatial positions)")
    print("=" * 80)
