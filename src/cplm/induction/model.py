from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class InductionModelConfig:
    vocab_size: int
    d_model: int = 96
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 192
    max_seq_len: int = 128
    dropout: float = 0.0
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn_drop = self.dropout(attn)
        y = attn_drop @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.out_proj(y)
        return y, (attn if return_attn else None)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: InductionModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        ) if cfg.d_ff and cfg.d_ff > 0 else None

    def forward(self, x: torch.Tensor, *, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        a, attn = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + a
        if self.mlp is not None:
            x = x + self.mlp(self.ln_2(x))
        return x, attn


class InductionTransformer(nn.Module):
    def __init__(self, cfg: InductionModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, *, return_attn: bool = False, return_hidden: bool = False):
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        attns: List[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attns.append(attn)
        h = self.ln_f(x)
        logits = self.lm_head(h)
        out = {"logits": logits}
        if return_attn:
            out["attentions"] = attns
        if return_hidden:
            out["hidden"] = h
        return out

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self(input_ids)["logits"]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
