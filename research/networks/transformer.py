"""
Taken from nanoGPT by Andrej Karpathy
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
import math

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def transformer_weight_init(module: nn.Module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, n_embd, bias, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class MLP(nn.Module):
    def __init__(self, n_embd=128, dropout=0.0, dense_multiplier=4, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, dense_multiplier * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self, n_embd: int = 128, n_head: int = 4, dropout: float = 0.0, bias: bool = True, causal: bool = True
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # Causal
        self.causal = causal

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=self.causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int = 128,
        n_head: int = 4,
        dropout: float = 0.0,
        dense_multiplier: int = 4,
        bias: bool = False,
        causal: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias, eps=eps)
        self.attn = SelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout, bias=bias, causal=causal)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, dense_multiplier=dense_multiplier, bias=bias)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.0,
        dense_multiplier: int = 4,
        bias: bool = False,
        causal: bool = True,
        eps: float = 1e-5,
        block_size: int = 128,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    dense_multiplier=dense_multiplier,
                    bias=bias,
                    causal=causal,
                    eps=eps,
                )
                for _ in range(n_layer)
            ]
        )
        self.layer_norm = LayerNorm(n_embd, bias=bias, eps=eps)

        self.apply(transformer_weight_init)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, x, attn_mask=None):
        assert len(x.shape) == 3
        pos_idxs = torch.arange(0, x.shape[1], device=x.device, dtype=torch.long)
        x = x + self.pos_embedding(pos_idxs)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.layer_norm(x)
        return x


class TransformerStateSequenceEncoder(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, n_embd=128, **kwargs):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1
        self.n_embd = n_embd
        self.transformer = TransformerEncoder(n_embd=n_embd, **kwargs)
        self.obs_embedding = nn.Linear(observation_space.shape[0], n_embd)
        nn.init.normal_(self.obs_embedding.weight, mean=0.0, std=0.02)

    @property
    def output_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_embd,), dtype=np.float32)

    def forward(self, obs, mask=None):
        assert len(obs.shape) == 3
        return self.transformer(self.obs_embedding(obs), attn_mask=mask)
