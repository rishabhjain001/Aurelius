"""
encoders/text.py — Transformer Text Encoder for Aurelius CLIP-VLM

Architecture:
    Tokenizer (tiktoken GPT-2 BPE, truncate/pad to 77)
    → Token Embedding + Learned Positional Embedding
    → L × TransformerBlock (pre-norm, MHSA from scratch, FFN)
    → Mean pool (non-padded positions only)
    → ProjectionHead
    → L2 normalize

Implements:
    - Tokenizer (wrapper around tiktoken)
    - MultiHeadSelfAttention
    - TransformerBlock
    - TextEncoder

Requirements: 1.1, 1.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8,
              6.1, 6.6, 6.7, 25.2
"""

import math
from typing import List

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from encoders.image import LinearProjectionHead, MLPProjectionHead


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Wraps tiktoken GPT-2 BPE encoding with truncation and padding.

    Args:
        max_seq_len: Maximum sequence length (default 77).
        pad_id: Token ID used for padding positions (default 0).

    Notes:
        GPT-2 vocabulary is 0–50256. pad_id=0 is a real token ("!"),
        but the attention mask will exclude all pad positions, so the
        model never attends to them.  pad_id=50257 (outside vocab) is
        also a valid choice if a larger embedding table is used.
    """

    def __init__(self, max_seq_len: int = 77, pad_id: int = 0) -> None:
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        """Encode a single string → list of ints of length max_seq_len.

        Truncates if len > max_seq_len, pads with pad_id if len < max_seq_len.
        """
        ids = self._enc.encode(text)
        # Truncate
        ids = ids[: self.max_seq_len]
        # Pad
        ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        assert len(ids) == self.max_seq_len
        return ids


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (from scratch — no torch.nn.MultiheadAttention)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention built entirely from primitive nn.Linear layers.

    Args:
        d_model: Total model dimension.
        num_heads: Number of attention heads (d_model must be divisible).

    Notes:
        - Q, K, V projections: three separate nn.Linear(d_model, d_model, bias=False)
        - Output projection: nn.Linear(d_model, d_model, bias=False)
        - head_dim = d_model // num_heads
        - Attention score: (Q @ K^T) / sqrt(head_dim), then add additive mask, then softmax
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x:    [N, seq_len, d_model]
            mask: [N, 1, 1, seq_len] additive mask
                  (0.0 for real tokens, -1e9 for pad positions)

        Returns:
            [N, seq_len, d_model]
        """
        N, seq_len, _ = x.shape
        H = self.num_heads
        D_h = self.head_dim

        # Project to Q, K, V  — each [N, seq_len, d_model]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to [N, H, seq_len, D_h]
        Q = Q.view(N, seq_len, H, D_h).transpose(1, 2)  # [N, H, seq_len, D_h]
        K = K.view(N, seq_len, H, D_h).transpose(1, 2)
        V = V.view(N, seq_len, H, D_h).transpose(1, 2)

        # Attention scores: [N, H, seq_len, seq_len]
        scale = math.sqrt(D_h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [N, H, seq_len, seq_len]

        # Apply additive mask — mask is [N, 1, 1, seq_len], broadcast over heads and query positions
        scores = scores + mask  # -1e9 makes pad positions vanish after softmax

        # Softmax over key dimension
        attn_weights = torch.softmax(scores, dim=-1)  # [N, H, seq_len, seq_len]

        # Weighted sum over V
        out = torch.matmul(attn_weights, V)  # [N, H, seq_len, D_h]

        # Concatenate heads: [N, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.d_model)

        # Output projection
        out = self.out_proj(out)  # [N, seq_len, d_model]

        return out


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block.

    Sub-layer 1: LayerNorm → MultiHeadSelfAttention → residual add
    Sub-layer 2: LayerNorm → FFN → residual add

    FFN: Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)

    Args:
        d_model:          Model dimension.
        num_heads:        Number of attention heads.
        ffn_multiplier:   Hidden dimension multiplier for the FFN (default 4).
    """

    def __init__(self, d_model: int, num_heads: int, ffn_multiplier: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)

        self.norm2 = nn.LayerNorm(d_model)
        ffn_hidden = d_model * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x:    [N, seq_len, d_model]
            mask: [N, 1, 1, seq_len] additive padding mask

        Returns:
            [N, seq_len, d_model]
        """
        # Sub-layer 1: pre-norm MHSA + residual
        x = x + self.attn(self.norm1(x), mask)

        # Sub-layer 2: pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))

        return x


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Transformer-based text encoder for Aurelius CLIP-VLM.

    Takes a list of raw caption strings and returns L2-normalized embeddings.

    Architecture:
        1. Tokenize (tiktoken GPT-2 BPE, truncate/pad to max_seq_len)
        2. Token embedding + learned positional embedding
        3. L × TransformerBlock (pre-norm, MHSA from scratch)
        4. Mean pool over non-padded positions
        5. Projection head (linear or MLP)
        6. L2 normalize

    Args:
        config: Object with attributes:
            config.text_encoder.vocab_size      (int, 50257)
            config.text_encoder.max_seq_len     (int, 77)
            config.text_encoder.pad_id          (int, 0)
            config.text_encoder.d_model         (int, 512)
            config.text_encoder.num_heads       (int, 8)
            config.text_encoder.num_layers      (int, 6)
            config.text_encoder.ffn_multiplier  (int, 4)
            config.projection_head              ("linear" | "mlp")
            config.embed_dim                    (int, 256)

    All weights are initialized from scratch (no pretrained values).

    Note: torch.nn.MultiheadAttention, torch.nn.Transformer, and any
          RNN/LSTM/GRU are explicitly not used.
    """

    def __init__(self, config) -> None:
        super().__init__()

        te_cfg = config.text_encoder
        vocab_size: int = te_cfg.vocab_size
        max_seq_len: int = te_cfg.max_seq_len
        pad_id: int = te_cfg.pad_id
        d_model: int = te_cfg.d_model
        num_heads: int = te_cfg.num_heads
        num_layers: int = te_cfg.num_layers
        ffn_multiplier: int = te_cfg.ffn_multiplier
        proj_head_type: str = config.projection_head
        embed_dim: int = config.embed_dim

        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embed_dim = embed_dim

        # Tokenizer
        self.tokenizer = Tokenizer(max_seq_len=max_seq_len, pad_id=pad_id)

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding: [max_seq_len, d_model]
        self.positional_embedding = nn.Parameter(
            torch.zeros(max_seq_len, d_model)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_multiplier)
            for _ in range(num_layers)
        ])

        # Projection head
        if proj_head_type == "linear":
            self.projection_head = LinearProjectionHead(d_model, embed_dim)
        elif proj_head_type == "mlp":
            self.projection_head = MLPProjectionHead(d_model, embed_dim)
        else:
            raise ValueError(
                f"Unknown projection_head type '{proj_head_type}'. "
                "Expected 'linear' or 'mlp'."
            )

    def forward(self, captions: List[str]) -> Tensor:
        """Encode a batch of caption strings into L2-normalized embeddings.

        Args:
            captions: List of N caption strings (any length, including empty).

        Returns:
            Tensor of shape [N, embed_dim], L2-normalized.
        """
        N = len(captions)
        device = self.positional_embedding.device

        # ------------------------------------------------------------------
        # 1. Tokenize each caption → [N, max_seq_len] int tensor
        # ------------------------------------------------------------------
        token_ids = [self.tokenizer.encode(cap) for cap in captions]
        tokens = torch.tensor(token_ids, dtype=torch.long, device=device)  # [N, seq_len]

        # ------------------------------------------------------------------
        # 2. Build additive padding mask  [N, 1, 1, seq_len]
        #    0.0 for real tokens, -1e9 for pad positions
        # ------------------------------------------------------------------
        # is_pad: True where token == pad_id  [N, seq_len]
        is_pad = (tokens == self.pad_id)  # [N, seq_len]
        # Additive mask: 0 for real, -1e9 for pad
        attn_mask = torch.zeros(N, self.max_seq_len, dtype=torch.float32, device=device)
        attn_mask = attn_mask.masked_fill(is_pad, -1e9)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, seq_len]

        # ------------------------------------------------------------------
        # 3. Embeddings: token + positional
        # ------------------------------------------------------------------
        x = self.token_embedding(tokens)            # [N, seq_len, d_model]
        x = x + self.positional_embedding.unsqueeze(0)  # broadcast [1, seq_len, d_model]

        # ------------------------------------------------------------------
        # 4. Transformer blocks
        # ------------------------------------------------------------------
        for block in self.blocks:
            x = block(x, attn_mask)  # [N, seq_len, d_model]

        # ------------------------------------------------------------------
        # 5. Mean pool over non-padded positions
        #    Zero out pad positions before averaging.
        # ------------------------------------------------------------------
        # real_mask: 1.0 for real tokens, 0.0 for pad  [N, seq_len, 1]
        real_mask = (~is_pad).float().unsqueeze(-1)  # [N, seq_len, 1]
        # Sum over real positions, divide by count of real positions
        x = x * real_mask                                         # zero out pads
        token_counts = real_mask.sum(dim=1).clamp(min=1.0)       # [N, 1]
        pooled = x.sum(dim=1) / token_counts                     # [N, d_model]

        # ------------------------------------------------------------------
        # 6. Projection head
        # ------------------------------------------------------------------
        out = self.projection_head(pooled)  # [N, embed_dim]

        # ------------------------------------------------------------------
        # 7. L2 normalize
        # ------------------------------------------------------------------
        out = F.normalize(out, p=2, dim=-1)  # [N, embed_dim]

        return out
