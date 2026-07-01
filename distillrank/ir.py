"""PyTorch low-rank IR: replace target nn.Linear with a trainable two-matrix form.

LowRankLinear(x) = up(down(x)), down: in->r (no bias), up: r->out (+orig bias).
Initialized from a (plain or activation-aware) SVD of the original weight, then
finetuned. Export maps each module's factors straight to the runtime tensors:
svd_vt = down.weight [r,in], svd_u = up.weight [out,r], svd_s = ones(r).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .calibrate import gguf_name
from .factorize import RankPolicy, saves_params, whiten_svd, _svd


class LowRankLinear(nn.Module):
    def __init__(self, up_w: torch.Tensor, down_w: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        out, r = up_w.shape
        _r, in_ = down_w.shape
        self.down = nn.Linear(in_, r, bias=False)
        self.up = nn.Linear(r, out, bias=bias is not None)
        with torch.no_grad():
            self.down.weight.copy_(down_w)
            self.up.weight.copy_(up_w)
            if bias is not None:
                self.up.bias.copy_(bias)

    def forward(self, x):
        return self.up(self.down(x))


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    *path, last = name.split(".")
    parent = root
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, last, new)


def _factor(W: np.ndarray, H: np.ndarray | None, policy: RankPolicy):
    """Return (up[out,r], down[r,in]). Split the singular values symmetrically
    (√s into each factor) so the two matrices are balanced for finetuning."""
    if H is not None:
        U, s, Vt, _ = whiten_svd(W, H, policy)
    else:
        Wc = np.ascontiguousarray(W, dtype=np.float32)
        U0, s0, Vt0 = _svd(Wc)
        r = policy.choose(s0, W.shape[0], W.shape[1])
        U, s, Vt = U0[:, :r], s0[:r], Vt0[:r, :]
    rs = np.sqrt(np.maximum(s, 0)).astype(np.float32)
    return (U * rs).astype(np.float32), (rs[:, None] * Vt).astype(np.float32)


def make_lowrank(model: nn.Module, policy: RankPolicy, stats: dict | None = None) -> list[str]:
    """Replace each block linear (that passes the break-even guard) with a LowRankLinear."""
    done = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        g = gguf_name(name)
        if g is None or not g.startswith("blk."):   # skip lm_head/embeddings for now
            continue
        W = mod.weight.detach().to(torch.float32).numpy()
        out, in_ = W.shape
        H = stats.get(g) if stats else None
        up, down = _factor(W, H, policy)
        if not saves_params(up.shape[1], out, in_):
            continue
        bias = mod.bias.detach().clone() if mod.bias is not None else None
        _set_module(model, name, LowRankLinear(torch.from_numpy(up), torch.from_numpy(down), bias))
        done.append(g)
    return done


def _permute_rows(w: np.ndarray, n_head: int) -> np.ndarray:
    """llama.cpp's q/k RoPE permutation, applied to the output (row) dimension.
    Since GGUF stores permuted q/k, and W_gguf = P·W = P·(up@down) = (P·up)@down,
    we apply it to the `up` factor's rows only."""
    out = w.shape[0]
    return (w.reshape(n_head, 2, out // n_head // 2, *w.shape[1:])
            .swapaxes(1, 2).reshape(w.shape))


def extract_factors(model: nn.Module) -> dict:
    """{gguf_base: (svd_u[out,r], svd_s ones[r], svd_vt[r,in])} for each LowRankLinear.

    q/k factors are permuted to GGUF space to match convert_hf_to_gguf (RoPE)."""
    cfg = model.config
    n_head = getattr(cfg, "num_attention_heads", None)
    n_kv = getattr(cfg, "num_key_value_heads", n_head)
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LowRankLinear):
            g = gguf_name(name)
            U = np.ascontiguousarray(mod.up.weight.detach().to(torch.float32).numpy())
            Vt = np.ascontiguousarray(mod.down.weight.detach().to(torch.float32).numpy())
            if g.endswith(".attn_q") and n_head:
                U = np.ascontiguousarray(_permute_rows(U, n_head))
            elif g.endswith(".attn_k") and n_kv:
                U = np.ascontiguousarray(_permute_rows(U, n_kv))
            out[g] = (U, np.ones(U.shape[1], dtype=np.float32), Vt)
    return out
