#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
FoldingDiT 3B Standalone - Complete Self-Contained Implementation

This file contains ALL the necessary components for the 3B parameter FoldingDiT model,
including the ESM-2 encoder and the full diffusion architecture. No external imports
from the simplefold package are needed (except torch and basic libraries).

**Note**: This file does NOT include the training loop. For training, see:
- src/simplefold/training/train.py (main training script)
- src/simplefold/model/simplefold.py (PyTorch Lightning training wrapper)

═══════════════════════════════════════════════════════════════════════════════
                         ESM-2 3B USAGE FLOW DIAGRAM
═══════════════════════════════════════════════════════════════════════════════

PREPROCESSING (Done once per protein, cached):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Input: "MKTAYIAKQRGHGKKSA..."  (amino acid sequence)                       │
│                                                                             │
│ ┌─────────────────────────────────────────────────┐                        │
│ │  compute_language_model_representations()       │                        │
│ │  ↓                                               │                        │
│ │  ESM-2 3B Transformer (36 layers, 2560 dim)     │  ← 3B parameters      │
│ │  ↓                                               │                        │
│ │  Extract all 37 layer outputs                   │                        │
│ └─────────────────────────────────────────────────┘                        │
│                        ↓                                                    │
│ Output: batch['esm_s'] = [B, M, 37, 2560]  ← CACHED IN DATASET            │
└─────────────────────────────────────────────────────────────────────────────┘

TRAINING (Uses cached ESM embeddings):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Input: noisy_coords [B,N,3], timestep t, batch (with cached esm_s)         │
│                                                                             │
│ FoldingDiT.forward():                                                       │
│   1. Atom Encoder (4 blocks, 640 dim)                                      │
│      ↓                                                                      │
│   2. Pool atoms → residues [B, M, 2048]                                    │
│      ↓                                                                      │
│   3. ╔═══════════════════════════════════════╗                             │
│      ║ ESM INTEGRATION (line 558-580):      ║  ← ESM USED HERE!           │
│      ║ - Load cached batch['esm_s']          ║                             │
│      ║ - Learned combination of 37 layers    ║                             │
│      ║ - Project 2560 → 2048                 ║                             │
│      ║ - Concatenate with geometric features ║                             │
│      ╚═══════════════════════════════════════╝                             │
│      ↓                                                                      │
│   4. Residue Trunk (36 blocks, 2048 dim)                                   │
│      ↓                                                                      │
│   5. Broadcast residues → atoms                                            │
│      ↓                                                                      │
│   6. Atom Decoder (4 blocks, 640 dim)                                      │
│      ↓                                                                      │
│ Output: predicted_velocity [B, N, 3]                                       │
└─────────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: ESM runs ONCE per protein (preprocessing), then embeddings are
             reused for THOUSANDS of training steps!
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

# ============================================================================
# 1. Residue Constants (from simplefold.utils.residue_constants)
# ============================================================================

restypes = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]
restypes_with_x = restypes + ["X"]

# ============================================================================
# 2. ESM Utils (from simplefold.utils.esm_utils)
# ============================================================================

load_fn = torch.hub.load
esm_registry = {
    "esm2_8M": partial(load_fn, "facebookresearch/esm:main", "esm2_t6_8M_UR50D"),
    "esm2_35M": partial(load_fn, "facebookresearch/esm:main", "esm2_t12_35M_UR50D"),
    "esm2_150M": partial(load_fn, "facebookresearch/esm:main", "esm2_t30_150M_UR50D"),
    "esm2_650M": partial(load_fn, "facebookresearch/esm:main", "esm2_t33_650M_UR50D"),
    "esm2_3B": partial(load_fn, "facebookresearch/esm:main", "esm2_t36_3B_UR50D"),
    "esm2_15B": partial(load_fn, "facebookresearch/esm:main", "esm2_t48_15B_UR50D"),
}

esm_model_dict = {
    "esm2_8M": {"esm_s_dim": 320, "esm_z_dim": 120, "esm_num_layers": 7},
    "esm2_35M": {"esm_s_dim": 480, "esm_z_dim": 240, "esm_num_layers": 13},
    "esm2_150M": {"esm_s_dim": 640, "esm_z_dim": 600, "esm_num_layers": 31},
    "esm2_650M": {"esm_s_dim": 1280, "esm_z_dim": 660, "esm_num_layers": 34},
    "esm2_3B": {"esm_s_dim": 2560, "esm_z_dim": 1440, "esm_num_layers": 37},
    "esm2_15B": {"esm_s_dim": 5120, "esm_z_dim": 1920, "esm_num_layers": 49},
}

def _af2_to_esm(d):
    """Convert AlphaFold2 residue indices to ESM indices."""
    esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in restypes_with_x]
    return torch.tensor(esm_reorder)

def compute_language_model_representations(esmaa, esm, esm_dict, backend="torch"):
    """
    Run ESM-2 forward pass to extract per-residue embeddings from all layers.
    
    This is THE KEY FUNCTION where ESM-2 is actually used!
    
    Args:
        esmaa: [B, L] - Amino acid sequence indices (ESM format)
        esm: ESM-2 model (e.g., 3B parameter transformer)
        esm_dict: ESM dictionary with special tokens
        backend: "torch" or "mlx"
    
    Returns:
        esm_s: [B, L, num_layers+1, esm_dim] - Embeddings from all ESM layers
               For ESM-2 3B: [B, L, 37, 2560]
               
    ESM-2 Architecture (3B):
    - 36 transformer layers + 1 embedding layer = 37 total layers
    - Each layer outputs 2560-dimensional embeddings
    - Input: Amino acid sequence "MKTAYIAKQR..." 
    - Output: Contextualized embeddings capturing evolutionary patterns
    
    Usage in FoldingDiT:
    1. This function is called during data preprocessing (NOT in forward pass)
    2. Results are cached in batch['esm_s']
    3. FoldingDiT.forward() uses these precomputed embeddings
    4. They're combined with geometric features via learned projection
    """
    batch_size = esmaa.size(0)
    
    # Add BOS/EOS tokens required by ESM
    bosi, eosi = esm_dict.cls_idx, esm_dict.eos_idx
    bos = esmaa.new_full((batch_size, 1), bosi)
    eos = esmaa.new_full((batch_size, 1), esm_dict.padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi
    
    if backend == "mlx":
        import mlx.core as mx
        esmaa = mx.array(esmaa)
    
    # **THIS IS WHERE ESM-2 RUNS**: Forward pass through 3B parameter model
    res = esm(
        esmaa,
        repr_layers=range(esm.num_layers + 1),  # Extract all 37 layers
        need_head_weights=False,
    )
    
    if backend == "mlx":
        import numpy as np
        res['representations'] = {k: torch.from_numpy(np.array(v)) for k,v in res['representations'].items()}
    
    # Stack all layer outputs: [B, L, num_layers, dim]
    esm_s = torch.stack([v for _, v in sorted(res["representations"].items())], dim=2)
    esm_s = esm_s[:, 1:-1]  # Remove BOS/EOS tokens
    return esm_s, None

# ============================================================================
# 3. Layers (from simplefold.model.torch.layers)
# ============================================================================

def modulate(x, shift, scale):
    """Adaptive Layer Norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed

class SelfAttentionLayer(nn.Module):
    """Standard multi-head self-attention with RMS normalization."""
    def __init__(self, hidden_size, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0.0, proj_drop=0.0, use_bias=True, qk_norm=True, 
                 pos_embedder=None, linear_target=nn.Linear):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = linear_target(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_target(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.pos_embedder = pos_embedder

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        pos = kwargs.get("pos")
        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EfficientSelfAttentionLayer(SelfAttentionLayer):
    """Efficient self-attention using PyTorch's scaled_dot_product_attention."""
    def forward(self, x, **kwargs):
        B, N, C = x.shape
        attn_mask = kwargs.get("attention_mask")
        pos = kwargs.get("pos")
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = qkv.unbind(0)
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=q.dtype)
        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwiGLUFeedForward(nn.Module):
    """SwiGLU activation-based feedforward network."""
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations using sinusoidal encoding."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class ConditionEmbedder(nn.Module):
    """Embeds condition vectors with optional dropout for classifier-free guidance."""
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.dropout_prob = dropout_prob
        self.null_token = nn.Parameter(torch.randn(input_dim), requires_grad=True)

    def forward(self, cond, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            drop_ids = force_drop_ids if force_drop_ids is not None else (torch.rand(cond.shape[0], device=cond.device) < self.dropout_prob)
            cond[drop_ids] = self.null_token[None, None, :]
        return self.proj(cond)

class FinalLayer(nn.Module):
    """Final DiT layer with adaptive layer normalization."""
    def __init__(self, hidden_size, out_channels, c_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_dim, 2 * hidden_size, bias=True))
        # Zero initialization
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)

# ============================================================================
# 4. Position Embeddings (from simplefold.model.torch.pos_embed)
# ============================================================================

class AbsolutePositionEncoding(nn.Module):
    """Absolute sinusoidal position encoding."""
    def __init__(self, in_dim, embed_dim, include_input=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = embed_dim
        self.include_input = include_input
        self.embed_dim = embed_dim + in_dim if include_input else embed_dim

    def get_1d_pos_embed(self, pos):
        embed_dim = self.hidden_dim // (self.in_dim * 2)
        omega = 2 ** torch.linspace(0, math.log(224, 2) - 1, embed_dim).to(pos.device) * torch.pi
        if len(pos.shape) == 1:
            out = torch.einsum("m,d->md", pos, omega)
        elif len(pos.shape) == 2:
            out = torch.einsum("nm,d->nmd", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    def forward(self, pos):
        pos_embs = [self.get_1d_pos_embed(pos[..., i]) for i in range(self.in_dim)]
        if self.include_input:
            pos_embs.append(pos)
        return torch.cat(pos_embs, dim=-1)

class FourierPositionEncoding(nn.Module):
    """Fourier feature position encoding."""
    def __init__(self, in_dim, include_input=False, min_freq_log2=0, max_freq_log2=12, num_freqs=32, log_sampling=True):
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input
        self.num_freqs = num_freqs
        if log_sampling:
            self.register_buffer("freq_bands", 2.0 ** torch.linspace(min_freq_log2, max_freq_log2, steps=num_freqs))
        else:
            self.register_buffer("freq_bands", torch.linspace(2.0**min_freq_log2, 2.0**max_freq_log2, steps=num_freqs))
        self.embed_dim = (in_dim if include_input else 0) + in_dim * num_freqs * 2

    def forward(self, pos):
        out = [pos] if self.include_input else []
        pos_expanded = pos.unsqueeze(-1) * self.freq_bands
        out.extend([torch.sin(pos_expanded).flatten(start_dim=-2), torch.cos(pos_expanded).flatten(start_dim=-2)])
        return torch.cat(out, dim=-1)

def compute_axial_cis(ts, in_dim, dim, theta=100.0):
    """Compute axial complex frequency tensor for rotary embeddings."""
    B, N, D = ts.shape
    freqs_all = []
    interval = 2 * in_dim
    for i in range(in_dim):
        freq = 1.0 / (theta ** (torch.arange(0, dim, interval)[: (dim // interval)].float() / dim)).to(ts.device)
        t = ts[..., i].flatten()
        freq_i = torch.outer(t, freq)
        freq_cis_i = torch.polar(torch.ones_like(freq_i), freq_i).view(B, N, -1)
        freqs_all.append(freq_cis_i)
    return torch.cat(freqs_all, dim=-1)

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary position embeddings to queries and keys."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class AxialRotaryPositionEncoding(nn.Module):
    """Axial Rotary Position Encoding (RoPE) for 3D+chain coordinates."""
    def __init__(self, in_dim, embed_dim, num_heads, base=100.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim // num_heads
        self.base = base

    def forward(self, xq, xk, pos):
        if pos.ndim == 2: 
            pos = pos.unsqueeze(-1)
        freqs_cis = compute_axial_cis(pos, self.in_dim, self.embed_dim, self.base).unsqueeze(1)
        return apply_rotary_emb(xq, xk, freqs_cis.to(xq.device))

# ============================================================================
# 5. Blocks (from simplefold.model.torch.blocks)
# ============================================================================

class Mlp(nn.Module):
    """Standard MLP for non-SwiGLU blocks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    def __init__(self, self_attention_layer, hidden_size, mlp_ratio=4.0, use_swiglu=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            self.mlp = Mlp(hidden_size, mlp_hidden_dim, act_layer=partial(nn.GELU, approximate="tanh"))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        # Zero-out adaLN modulation
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, latents, c, **kwargs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        _latents = self.attn(modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs)
        latents = latents + gate_msa.unsqueeze(1) * _latents
        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(latents), shift_mlp, scale_mlp))
        return latents

class HomogenTrunk(nn.Module):
    """Homogeneous trunk of repeated identical blocks."""
    def __init__(self, block, depth):
        super().__init__()
        self.blocks = nn.ModuleList([block() for _ in range(depth)])

    def forward(self, latents, c, **kwargs):
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents

# ============================================================================
# 6. Architecture (from simplefold.model.torch.architecture)
# ============================================================================

class FoldingDiT(nn.Module):
    """
    FoldingDiT: Diffusion Transformer for Protein Structure Prediction
    
    Architecture Flow:
    1. Atom Features (N atoms) -> Atom Encoder (4 blocks) -> Atom Latents
    2. Atom Latents -> Pool to Residue Tokens (M residues)
    3. Residue Tokens + ESM Embeddings -> Residue Trunk (36 blocks for 3B)
    4. Residue Latents -> Broadcast to Atoms
    5. Atom Latents -> Atom Decoder (4 blocks) -> Predicted Velocities
    """
    def __init__(
        self, trunk, time_embedder, aminoacid_pos_embedder, pos_embedder,
        atom_encoder_transformer, atom_decoder_transformer,
        hidden_size=1152, num_heads=16, atom_num_heads=4, output_channels=3,
        atom_hidden_size_enc=256, atom_hidden_size_dec=256,
        atom_n_queries_enc=32, atom_n_keys_enc=128,
        atom_n_queries_dec=32, atom_n_keys_dec=128,
        esm_model="esm2_3B", esm_dropout_prob=0.0,
        use_atom_mask=False, use_length_condition=True,
    ):
        super().__init__()
        self.pos_embedder = pos_embedder
        pos_embed_channels = pos_embedder.embed_dim
        self.aminoacid_pos_embedder = aminoacid_pos_embedder
        aminoacid_pos_embed_channels = aminoacid_pos_embedder.embed_dim
        self.time_embedder = time_embedder
        self.atom_encoder_transformer = atom_encoder_transformer
        self.atom_decoder_transformer = atom_decoder_transformer
        self.trunk = trunk
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.atom_num_heads = atom_num_heads
        self.use_atom_mask = use_atom_mask
        self.esm_dropout_prob = esm_dropout_prob
        self.use_length_condition = use_length_condition
        self.atom_hidden_size_enc = atom_hidden_size_enc
        self.atom_hidden_size_dec = atom_hidden_size_dec
        self.atom_n_queries_enc = atom_n_queries_enc
        self.atom_n_keys_enc = atom_n_keys_enc
        self.atom_n_queries_dec = atom_n_queries_dec
        self.atom_n_keys_dec = atom_n_keys_dec

        esm_s_dim = esm_model_dict[esm_model]["esm_s_dim"]
        esm_num_layers = esm_model_dict[esm_model]["esm_num_layers"]

        # Atom feature projection: pos_embed + aa_pos_embed + 427 features
        atom_feat_dim = pos_embed_channels + aminoacid_pos_embed_channels + 427
        self.atom_feat_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.atom_pos_proj = nn.Linear(pos_embed_channels, hidden_size, bias=False)

        if self.use_length_condition:
            self.length_embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
            )

        self.atom_in_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # ESM embedding combination and projection
        self.esm_s_combine = nn.Parameter(torch.zeros(esm_num_layers))
        self.esm_s_proj = ConditionEmbedder(esm_s_dim, hidden_size, self.esm_dropout_prob)
        self.esm_cat_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Atom-to-residue and residue-to-atom projections
        self.context2atom_proj = nn.Sequential(nn.Linear(hidden_size, self.atom_hidden_size_enc), nn.LayerNorm(self.atom_hidden_size_enc))
        self.atom2latent_proj = nn.Sequential(nn.Linear(self.atom_hidden_size_enc, hidden_size), nn.LayerNorm(hidden_size))
        self.atom_enc_cond_proj = nn.Sequential(nn.Linear(hidden_size, self.atom_hidden_size_enc), nn.LayerNorm(self.atom_hidden_size_enc))
        self.atom_dec_cond_proj = nn.Sequential(nn.Linear(hidden_size, self.atom_hidden_size_dec), nn.LayerNorm(self.atom_hidden_size_dec))
        self.latent2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.atom_hidden_size_dec)
        )
        self.final_layer = FinalLayer(self.atom_hidden_size_dec, output_channels, c_dim=hidden_size)

    def create_local_attn_bias(self, n, n_queries, n_keys, inf=1e10, device=None):
        """Create local attention bias for sliding window attention."""
        n_trunks = int(math.ceil(n / n_queries))
        attn_mask = torch.zeros(n_trunks * n_queries, n_trunks * n_queries, device=device)
        for block_index in range(0, n_trunks):
            i = block_index * n_queries
            j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
            j2 = n_queries * block_index + (n_queries + n_keys) // 2
            attn_mask[i : i + n_queries, j1:j2] = 1.0
        return ((1 - attn_mask) * -inf).to(device=device)[:n, :n]

    def create_atom_attn_mask(self, feats, natoms, atom_n_queries=None, atom_n_keys=None, inf=1e10):
        """Create attention mask for atom encoder/decoder."""
        if atom_n_queries is not None and atom_n_keys is not None:
            return self.create_local_attn_bias(natoms, atom_n_queries, atom_n_keys, device=feats["ref_pos"].device, inf=inf)
        return None

    def forward(self, noised_pos, t, feats, self_cond=None):
        """
        Forward pass through FoldingDiT.
        
        Args:
            noised_pos: [B, N, 3] - Noisy atom coordinates
            t: [B] - Diffusion timestep
            feats: dict - Feature dictionary containing:
                - ref_pos, mol_type, res_type, pocket_feature, etc.
                - esm_s: [B, M, num_layers, esm_dim] - ESM embeddings
        
        Returns:
            dict with keys:
                - predict_velocity: [B, N, 3] - Predicted denoising velocity
                - latent: [B, M, hidden_size] - Residue-level latents
        """
        B, N, _ = feats["ref_pos"].shape
        M = feats["mol_type"].shape[1]
        atom_to_token = feats["atom_to_token"].float()  # [B, N, M]
        atom_to_token_idx = feats["atom_to_token_idx"]
        ref_space_uid = feats["ref_space_uid"]

        # Create attention masks for local attention
        atom_attn_mask_enc = self.create_atom_attn_mask(feats, N, self.atom_n_queries_enc, self.atom_n_keys_enc)
        atom_attn_mask_dec = self.create_atom_attn_mask(feats, N, self.atom_n_queries_dec, self.atom_n_keys_dec)

        # Time and length conditioning
        c_emb = self.time_embedder(t)  # [B, D]
        if self.use_length_condition:
            length = feats["max_num_tokens"].float().unsqueeze(-1)
            c_emb = c_emb + self.length_embedder(torch.log(length))

        # Build atom features from residue features
        mol_type = F.one_hot(feats["mol_type"], num_classes=4).float()  # [B, M, 4]
        res_feat = torch.cat([mol_type, feats["res_type"].float(), feats["pocket_feature"].float()], dim=-1)  # [B, M, 41]
        atom_feat_from_res = torch.bmm(atom_to_token, res_feat)  # [B, N, 41]
        atom_res_pos = self.aminoacid_pos_embedder(pos=atom_to_token_idx.unsqueeze(-1).float())
        ref_pos_emb = self.pos_embedder(pos=feats["ref_pos"])
        
        # Concatenate all atom features
        atom_feat = torch.cat([
            ref_pos_emb, atom_feat_from_res, atom_res_pos,
            feats["ref_charge"].unsqueeze(-1), feats["atom_pad_mask"].unsqueeze(-1),
            feats["ref_element"], feats["ref_atom_name_chars"].reshape(B, N, 4 * 64),
        ], dim=-1)
        atom_feat = self.atom_feat_proj(atom_feat)  # [B, N, D]
        
        # Add noised coordinate information
        atom_coord = self.atom_pos_proj(self.pos_embedder(pos=noised_pos))  # [B, N, D]
        atom_in = self.atom_in_proj(torch.cat([atom_feat, atom_coord], dim=-1))  # [B, N, D]

        # Position embeddings for RoPE
        atom_pe_pos = torch.cat([ref_space_uid.unsqueeze(-1).float(), feats["ref_pos"]], dim=-1)  # [B, N, 4]
        token_pe_pos = torch.cat([
            feats["residue_index"].unsqueeze(-1).float(), feats["entity_id"].unsqueeze(-1).float(),
            feats["asym_id"].unsqueeze(-1).float(), feats["sym_id"].unsqueeze(-1).float(),
        ], dim=-1)  # [B, M, 4]

        # ATOM ENCODER: Process individual atoms
        atom_c_emb_enc = self.atom_enc_cond_proj(c_emb)
        atom_latent = self.atom2latent_proj(self.atom_encoder_transformer(
            latents=self.context2atom_proj(atom_in), 
            c=atom_c_emb_enc, 
            attention_mask=atom_attn_mask_enc,
            pos=atom_pe_pos,
        ))  # [B, N, D]

        # POOLING: Aggregate atom tokens to residue tokens
        atom_to_token_mean = atom_to_token / (atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
        latent = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_latent)  # [B, M, D]
        
        # ═══════════════════════════════════════════════════════════════════
        # ESM EMBEDDING INTEGRATION - This is where ESM-2 3B is actually used!
        # ═══════════════════════════════════════════════════════════════════
        # 
        # Input: feats['esm_s'] = [B, M, 37, 2560]
        #   - Precomputed by calling compute_language_model_representations()
        #   - Contains embeddings from all 37 ESM-2 layers
        #   - Each layer captures different evolutionary patterns
        #
        # Step 1: Learned layer combination
        # self.esm_s_combine is a learnable weight [37] that combines all layers
        # This learns which ESM layers are most useful for structure prediction
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ feats['esm_s']).squeeze(2)  # [B, M, 2560]
        
        # Step 2: Project ESM embeddings to model hidden dim with dropout
        # ConditionEmbedder: 2560 → 2048 with optional dropout for classifier-free guidance
        esm_emb = self.esm_s_proj(esm_s, self.training, feats.get("force_drop_ids", None))  # [B, M, 2048]
        
        # Step 3: Concatenate geometric features with ESM sequence features
        # latent = from atom features (geometry, chemistry)
        # esm_emb = from ESM-2 (evolutionary sequence patterns)
        latent = self.esm_cat_proj(torch.cat([latent, esm_emb], dim=-1))  # [B, M, D]
        # 
        # Why this matters:
        # - ESM-2 provides sequence context: "This is a helix" "This binds ATP"
        # - Geometric features provide current structure: "These atoms are here"
        # - Together they enable: "Move atoms to form the helix that ESM predicts"
        # ═══════════════════════════════════════════════════════════════════

        # RESIDUE TRUNK: Main transformer processing
        latent = self.trunk(latents=latent, c=c_emb, attention_mask=None, pos=token_pe_pos)  # [B, M, D]

        # BROADCASTING: Residue tokens back to atom tokens
        output = torch.bmm(atom_to_token, latent) + atom_latent  # [B, N, D] with skip connection
        output = self.latent2atom_proj(output)  # [B, N, atom_hidden_size_dec]
        
        # ATOM DECODER: Final atom-level processing
        atom_c_emb_dec = self.atom_dec_cond_proj(c_emb)
        output = self.atom_decoder_transformer(
            latents=output, c=atom_c_emb_dec,
            attention_mask=atom_attn_mask_dec, pos=atom_pe_pos
        )
        output = self.final_layer(output, c=c_emb)  # [B, N, 3]

        return {"predict_velocity": output, "latent": latent}

# ============================================================================
# 7. Standalone 3B Model Wrapper
# ============================================================================

class FoldingDiT3B(nn.Module):
    """
    Complete Self-Contained 3B FoldingDiT Model
    
    This class provides a fully hardcoded instantiation of the 3B parameter
    FoldingDiT model with all components explicitly defined.
    
    Architecture Overview:
    ----------------------
    Total Parameters: ~3.7B
    - ESM-2 3B Encoder: ~3B parameters (2560 dim, 37 layers)
    - FoldingDiT Core: ~700M parameters
      - Atom Encoder: 4 blocks × 640 hidden dim
      - Residue Trunk: 36 blocks × 2048 hidden dim  
      - Atom Decoder: 4 blocks × 640 hidden dim
    
    Forward Pass Flow:
    -----------------
    1. Input: Noisy atom coordinates [B, N, 3] + timestep t
    2. **ESM-2 PREPROCESSING** (happens BEFORE forward, cached in batch):
       - Amino acid sequence → ESM-2 3B → [B, M, 37, 2560] embeddings
       - Called via: compute_language_model_representations(sequence, esm_model, esm_dict)
       - Stores all 37 transformer layer outputs
    3. Atom Encoder: Processes N atoms → atom latents [B, N, 640]
    4. Pooling: Aggregate atoms to M residues [B, M, 2048]
    5. **ESM INTEGRATION** (happens IN forward):
       - Learned combination of 37 ESM layers → [B, M, 2560]
       - Project to hidden dim → [B, M, 2048]
       - Concatenate with geometric features
    6. Residue Trunk: 36-layer transformer on residues [B, M, 2048]
    7. Broadcasting: Residues back to atoms [B, N, 640]
    8. Atom Decoder: Final processing → velocity [B, N, 3]
    
    ESM-2 3B Usage Summary:
    ----------------------
    - WHEN: During data preprocessing (not in training forward pass)
    - WHERE: compute_language_model_representations() function
    - INPUT: "MKTAYIAKQRGHGKKSA..." (amino acid sequence)
    - OUTPUT: [B, M, 37, 2560] (per-residue embeddings from all layers)
    - STORAGE: Cached in batch['esm_s'] to avoid recomputation
    - USAGE IN TRAINING: FoldingDiT.forward() combines these with geometry
    7. Broadcasting: Residues back to atoms [B, N, 640]
    8. Atom Decoder: Final processing → velocity [B, N, 3]
    
    **IMPORTANT**: This file does NOT include the training loop.
    For training, see:
    - src/simplefold/training/train.py (training script)
    - src/simplefold/model/simplefold.py (Lightning wrapper with flow_matching_train_step)
    """
    def __init__(self, device="cpu"):
        super().__init__()
        
        # Configuration hardcoded from foldingdit_3B.yaml
        self.config = {
            "hidden_size": 2048,
            "num_heads": 32,
            "atom_num_heads": 10,
            "output_channels": 3,
            "use_atom_mask": False,
            "use_length_condition": True,
            "esm_dropout_prob": 0.0,
            "esm_model_name": "esm2_3B",
            "trunk_depth": 36,
            "mlp_ratio": 4.0,
            "atom_encoder_depth": 4,
            "atom_decoder_depth": 4,
            "atom_hidden_size": 640,
            "atom_heads": 10,
            "atom_n_queries": 32,
            "atom_n_keys": 128,
        }
        
        print(f"Loading ESM-2 3B model ({self.config['esm_model_name']})...")
        self.esm_model, self.esm_dict = esm_registry[self.config["esm_model_name"]]()
        self.esm_model.eval()
        self.esm_model.to(device)
        self.af2_to_esm = _af2_to_esm(self.esm_dict).to(device)
        
        # ═══════════════════════════════════════════════════════════════════
        # NOTE: ESM Model Usage
        # ═══════════════════════════════════════════════════════════════════
        # The ESM model loaded above is NOT called during forward()!
        # 
        # Instead, ESM is used BEFORE training/inference to precompute embeddings:
        # 
        # 1. PREPROCESSING (once per protein):
        #    sequence = "MKTAYIAKQRGHGK..."
        #    esm_s = compute_language_model_representations(
        #        sequence, self.esm_model, self.esm_dict
        #    )  # [B, M, 37, 2560]
        # 
        # 2. CACHING:
        #    batch['esm_s'] = esm_s  # Store in dataset
        # 
        # 3. TRAINING:
        #    # ESM model NOT called - uses cached embeddings
        #    output = model.forward(batch, t, noisy_coords)
        #    # Inside forward: combines batch['esm_s'] with geometric features
        # 
        # This design:
        # ✓ Avoids recomputing expensive ESM forward passes every training step
        # ✓ Reduces GPU memory (ESM-2 3B needs ~12GB just for inference)
        # ✓ Enables faster training (no ESM backprop)
        # ═══════════════════════════════════════════════════════════════════
        
        # Build all components
        time_embedder = TimestepEmbedder(hidden_size=self.config["hidden_size"])
        aa_pos_embedder = AbsolutePositionEncoding(in_dim=1, embed_dim=self.config["hidden_size"], include_input=True)
        pos_embedder = FourierPositionEncoding(in_dim=3, include_input=True, min_freq_log2=0, max_freq_log2=12, num_freqs=128, log_sampling=True)
        
        # Main residue trunk: 36 blocks
        trunk_block_cls = partial(
            DiTBlock, hidden_size=self.config["hidden_size"], mlp_ratio=self.config["mlp_ratio"], use_swiglu=True,
            self_attention_layer=partial(
                EfficientSelfAttentionLayer, hidden_size=self.config["hidden_size"], num_heads=self.config["num_heads"], qk_norm=True,
                pos_embedder=AxialRotaryPositionEncoding(in_dim=4, embed_dim=self.config["hidden_size"], num_heads=self.config["num_heads"], base=100.0)
            )
        )
        trunk = HomogenTrunk(block=trunk_block_cls, depth=self.config["trunk_depth"])
        
        # Atom encoder/decoder: 4 blocks each
        atom_block_cls = partial(
            DiTBlock, hidden_size=self.config["atom_hidden_size"], mlp_ratio=4.0, use_swiglu=True,
            self_attention_layer=partial(
                EfficientSelfAttentionLayer, hidden_size=self.config["atom_hidden_size"], num_heads=self.config["atom_heads"], qk_norm=True,
                pos_embedder=AxialRotaryPositionEncoding(in_dim=4, embed_dim=self.config["atom_hidden_size"], num_heads=self.config["atom_heads"], base=100.0)
            )
        )
        atom_encoder = HomogenTrunk(block=atom_block_cls, depth=self.config["atom_encoder_depth"])
        atom_decoder = HomogenTrunk(block=atom_block_cls, depth=self.config["atom_decoder_depth"])

        # Assemble full FoldingDiT
        self.model = FoldingDiT(
            trunk=trunk, time_embedder=time_embedder, aminoacid_pos_embedder=aa_pos_embedder, pos_embedder=pos_embedder,
            atom_encoder_transformer=atom_encoder, atom_decoder_transformer=atom_decoder,
            hidden_size=self.config["hidden_size"], num_heads=self.config["num_heads"], atom_num_heads=self.config["atom_heads"],
            output_channels=self.config["output_channels"], atom_hidden_size_enc=self.config["atom_hidden_size"],
            atom_hidden_size_dec=self.config["atom_hidden_size"], atom_n_queries_enc=self.config["atom_n_queries"],
            atom_n_keys_enc=self.config["atom_n_keys"], atom_n_queries_dec=self.config["atom_n_queries"],
            atom_n_keys_dec=self.config["atom_n_keys"], esm_model=self.config["esm_model_name"],
            esm_dropout_prob=self.config["esm_dropout_prob"], use_atom_mask=self.config["use_atom_mask"],
            use_length_condition=self.config["use_length_condition"]
        )
        self.model.to(device)

    def forward(self, batch, t, noisy_coords):
        """
        Forward pass through the 3B model.
        
        Args:
            batch: Feature dictionary with 'esm_s', 'mol_type', 'ref_pos', etc.
            t: Timestep [B]
            noisy_coords: Noisy coordinates [B, N, 3]
        
        Returns:
            Predicted velocity [B, N, 3]
        """
        return self.model(noised_pos=noisy_coords, t=t, feats=batch)["predict_velocity"]


# ============================================================================
# 8. Training Information (NOT INCLUDED IN THIS FILE)
# ============================================================================

"""
TRAINING PIPELINE OVERVIEW
==========================

This standalone file contains the MODEL ARCHITECTURE only. For training, 
you need to use the training scripts which implement:

1. Flow Matching Training Loop (src/simplefold/model/simplefold.py):
   - flow_matching_train_step() method:
     * Sample random timesteps: t ~ Uniform(0, 1)
     * Add noise to ground truth coords: y_t = path.sample(t, y, noise)
     * Forward pass: predicted_velocity = model(y_t, t, batch)
     * Compute loss: MSE(predicted_velocity, true_velocity)
     * Optional smooth LDDT loss

2. Main Training Script (src/simplefold/training/train.py):
   - PyTorch Lightning trainer setup
   - Data loading via BoltzDataModule
   - FSDP for multi-GPU training
   - EMA model updates
   - Checkpointing & logging

3. Data Pipeline (src/simplefold/boltz_data_pipeline/):
   - BoltzTokenizer: Structure → tokens
   - BoltzFeaturizer: Extract geometric features
   - BoltzCropper: Crop to max tokens

4. Key Training Components:
   - Path: LinearPath for flow matching (y_t = t*y + (1-t)*noise)
   - Sampler: EMSampler for inference
   - Processor: ProteinDataProcessor for ESM embeddings & augmentation
   
Example Training Command:
-------------------------
python src/simplefold/train.py experiment=train_fsdp

See configs/experiment/train_fsdp.yaml for hyperparameters.
"""

if __name__ == "__main__":
    print("="*80)
    print("FoldingDiT 3B Standalone Model")
    print("="*80)
    print("\nThis file contains:")
    print("✓ Complete model architecture (all classes hardcoded)")
    print("✓ ESM-2 3B encoder integration")
    print("✓ Full diffusion transformer pipeline")
    print("\nThis file does NOT contain:")
    print("✗ Training loop (see src/simplefold/training/train.py)")
    print("✗ Data loading (see src/simplefold/boltz_data_pipeline/)")
    print("✗ Inference scripts (see src/simplefold/inference.py)")
    print("\n" + "="*80)
    
    print("\nAttempting to initialize model...")
    try:
        model = FoldingDiT3B(device="cpu")
        print("✓ Model initialized successfully!")
        print(f"\nConfig: {model.config}")
        
        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        folding_params = sum(p.numel() for p in model.model.parameters())
        esm_params = sum(p.numel() for p in model.esm_model.parameters())
        print(f"\nParameter breakdown:")
        print(f"  FoldingDiT: {folding_params:,}")
        print(f"  ESM-2 3B:   {esm_params:,}")
        print(f"  Total:      {total_params:,}")
        
        print("\n" + "="*80)
        print("ESM-2 USAGE EXPLANATION")
        print("="*80)
        print("""
The ESM-2 3B model is a protein language model (like BERT for proteins) that
provides rich evolutionary and structural context.

HOW IT'S USED:
-------------

1. PREPROCESSING (done once per protein, BEFORE training):
   
   from simplefold.model.folding_3b_standalone import compute_language_model_representations
   
   # Convert sequence to ESM indices
   sequence = "MKTAYIAKQRGHGKKVADSLTY..."  # Amino acid sequence
   esmaa = af2_idx_to_esm_idx(sequence_indices, mask, af2_to_esm)
   
   # Run ESM-2 3B forward pass (expensive! ~3B parameters)
   esm_s, _ = compute_language_model_representations(
       esmaa, 
       model.esm_model,      # 3B parameter transformer
       model.esm_dict
   )
   # Result: [B, M, 37, 2560]
   #   - M = number of residues
   #   - 37 = ESM layers (36 transformer + 1 embedding)
   #   - 2560 = ESM hidden dimension
   
   # Cache this in your dataset!
   batch['esm_s'] = esm_s

2. TRAINING (uses cached ESM embeddings):
   
   # No ESM forward pass here - just use cached embeddings
   output = model.forward(batch, t, noisy_coords)
   
   # Inside FoldingDiT.forward():
   # - Combines 37 ESM layers with learned weights
   # - Projects 2560 → 2048 dimensions  
   # - Concatenates with geometric features
   # - Processes through 36-layer residue trunk

WHY THIS DESIGN:
---------------
✓ ESM embeddings are deterministic (same input = same output)
✓ Running ESM every training step would be wasteful
✓ Caching saves ~12GB GPU memory and 10x compute time
✓ ESM stays frozen (no gradients) - only FoldingDiT trains

WHAT ESM PROVIDES:
-----------------
- Evolutionary patterns: "This region is conserved across species"
- Secondary structure hints: "This sequence forms an alpha helix"
- Functional context: "This is an ATP binding site"
- Long-range dependencies: "These distant residues interact"

This information guides the diffusion model to predict realistic structures.
        """)
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print("\nNote: ESM model download requires internet connection.")
