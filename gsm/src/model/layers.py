#
# Transformer Building Blocks for Molecular DiT
# Adapted from SimpleFold (Apple Inc.)
#

import math
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    """Apply affine transformation for AdaLN"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                            Positional Embeddings                              #
#################################################################################


class FourierPositionEmbedder(nn.Module):
    """
    Fourier feature position embedder for 3D coordinates.
    Maps (x, y, z) → high-dimensional feature vector.
    
    This is NOT a positional encoding (no sequence order information).
    It's a spatial encoding that maps 3D coordinates to high-dim features.
    """
    def __init__(self, embed_dim=258, max_freq=10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        # Frequency bands for Fourier features
        # embed_dim must be divisible by 6 (2 * 3 coords)
        assert embed_dim % 6 == 0, "embed_dim must be divisible by 6"
        num_bands = embed_dim // 6
        
        # Create frequency bands (log-spaced)
        freq_bands = torch.logspace(
            0, math.log2(max_freq), num_bands, base=2.0
        )
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, pos):
        """
        Args:
            pos: [B, N, 3] 3D positions
            
        Returns:
            embeddings: [B, N, embed_dim] Fourier features
        """
        # pos: [B, N, 3]
        # freq_bands: [num_bands]
        
        # Compute sin/cos for each coordinate and frequency
        # [B, N, 3, num_bands]
        pos_expanded = pos.unsqueeze(-1) * self.freq_bands
        
        # Concatenate sin and cos
        # [B, N, 3, num_bands, 2] → [B, N, embed_dim]
        embeddings = torch.cat([
            torch.sin(pos_expanded),
            torch.cos(pos_expanded),
        ], dim=-1)
        
        # Flatten last two dimensions
        B, N, _ = pos.shape
        embeddings = embeddings.reshape(B, N, -1)
        
        return embeddings


#################################################################################
#                            Attention Layers                                  #
#################################################################################


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))
    
    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
        rms_x = norm_x * (x.shape[-1] ** (-0.5))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


class EfficientSelfAttentionLayer(nn.Module):
    """
    Efficient self-attention WITHOUT positional encodings.
    Uses Flash Attention when available.
    
    NOTE: For molecules, we do NOT use RoPE or any positional encodings.
    Attention is purely based on learned features, not positions.
    """
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_bias=True,
        qk_norm=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
    
    def forward(self, x, **kwargs):
        B, N, C = x.shape
        attn_mask = kwargs.get("attention_mask")
        # NOTE: 'pos' kwarg is ignored - no positional encodings used!
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = qkv.unbind(0)
        
        # NO RoPE! Just normalize
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Attention mask preprocessing
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=q.dtype)
        
        # Use Flash Attention if available (faster!)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


#################################################################################
#                              FeedForward Layer                                #
#################################################################################


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feedforward network (from LLaMA).
    More expressive than standard MLP.
    """
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        if self.w2.bias is not None:
            nn.init.constant_(self.w2.bias, 0)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


#################################################################################
#                               Transformer Block                               #
#################################################################################


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with Adaptive Layer Norm (AdaLN-Zero).
    """
    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation (important for stable training)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, latents, c, **kwargs):
        """
        Args:
            latents: [B, N, hidden_size] input features
            c: [B, hidden_size] conditioning (timestep embedding)
            **kwargs: attention_mask, pos, etc.
        """
        # AdaLN modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        
        # Self-attention with residual
        _latents = self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa),
            **kwargs
        )
        latents = latents + gate_msa.unsqueeze(1) * _latents
        
        # MLP with residual
        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        
        return latents


class HomogenTrunk(nn.Module):
    """
    Stack of homogeneous transformer blocks.
    """
    def __init__(self, block, depth):
        super().__init__()
        self.blocks = nn.ModuleList([block() for _ in range(depth)])
    
    def forward(self, latents, c, **kwargs):
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents


#################################################################################
#                               Utility Layers                                  #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal features.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds conditioning signals (e.g., DreaMS embeddings) with optional dropout.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        self.dropout_prob = dropout_prob
    
    def forward(self, x, training=False, force_drop_ids=None):
        x = self.proj(x)
        
        if training and self.dropout_prob > 0:
            # Classifier-free guidance: drop entire samples
            if force_drop_ids is not None:
                x[force_drop_ids] = 0
            else:
                mask = torch.rand(x.shape[0], 1, device=x.device) > self.dropout_prob
                x = x * mask
        
        return x


class FinalLayer(nn.Module):
    """
    Final output layer of DiT with AdaLN.
    """
    def __init__(self, hidden_size, out_channels, c_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, 2 * hidden_size, bias=True)
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Zero-out output layers for stable initialization
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    print("Testing molecular DiT layers...")
    print("=" * 80)
    
    # Test Fourier embedder (embed_dim must be divisible by 6)
    pos_embedder = FourierPositionEmbedder(embed_dim=258)  # 258 = 43 * 6
    pos = torch.randn(2, 10, 3)  # Batch=2, N=10 atoms, 3D coords
    emb = pos_embedder(pos)
    print(f"✓ FourierPositionEmbedder: {pos.shape} → {emb.shape}")
    
    # Test attention (NO RoPE!)
    attn_layer = lambda: EfficientSelfAttentionLayer(
        hidden_size=512,
        num_heads=8,
    )
    attn = attn_layer()
    x = torch.randn(2, 10, 512)
    out = attn(x)  # No pos argument needed!
    print(f"✓ EfficientSelfAttentionLayer: {x.shape} → {out.shape}")
    
    # Test DiT block
    block_fn = lambda: DiTBlock(
        self_attention_layer=attn_layer,
        hidden_size=512,
        use_swiglu=True,
    )
    block = block_fn()
    c = torch.randn(2, 512)  # Conditioning
    out = block(x, c)  # No pos argument needed!
    print(f"✓ DiTBlock: {x.shape} → {out.shape}")
    
    # Test trunk
    trunk = HomogenTrunk(block_fn, depth=4)
    out = trunk(x, c)  # No pos argument needed!
    print(f"✓ HomogenTrunk (4 layers): {x.shape} → {out.shape}")
    
    # Test timestep embedder
    time_emb = TimestepEmbedder(hidden_size=512)
    t = torch.rand(2)
    t_emb = time_emb(t)
    print(f"✓ TimestepEmbedder: {t.shape} → {t_emb.shape}")
    
    # Test final layer
    final = FinalLayer(hidden_size=512, out_channels=3, c_dim=512)
    out = final(torch.randn(2, 10, 512), c)
    print(f"✓ FinalLayer: (2, 10, 512) → {out.shape}")
    
    print("=" * 80)
    print("✓ All layer tests passed!")
