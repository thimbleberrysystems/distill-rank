"""Global rank-budget allocation.

Instead of one rank/frac for every layer, pick a single energy threshold τ so the
whole model hits a target parameter ratio, letting each layer keep the smallest
rank that captures τ of its singular-value energy. Layers whose spectra decay
fast get compressed more — a simple, principled global allocation.
"""
from __future__ import annotations

import numpy as np

from . import ggufio
from .factorize import RankPolicy, saves_params

import torch


def _svdvals(W: np.ndarray, H: np.ndarray | None, G: np.ndarray | None = None) -> np.ndarray:
    """Singular values for the energy metric — whitened by H (right, input) and
    optionally G (left, output-gradient) so the threshold matches the export:
    svdvals(W·S) for activation-aware, svdvals(L·W·S) for two-sided (H=SSᵀ, G=LLᵀ)."""
    Wt = torch.tensor(np.ascontiguousarray(W), dtype=torch.float64)
    if G is not None:
        Gt = torch.tensor(np.ascontiguousarray(G), dtype=torch.float64)
        eps = 1e-3 * torch.diagonal(Gt).mean().clamp(min=1e-8)
        Wt = torch.linalg.cholesky(Gt + eps * torch.eye(Gt.shape[0], dtype=torch.float64)) @ Wt
    if H is not None:
        Ht = torch.tensor(np.ascontiguousarray(H), dtype=torch.float64)
        eps = 1e-3 * torch.diagonal(Ht).mean().clamp(min=1e-8)
        Wt = Wt @ torch.linalg.cholesky(Ht + eps * torch.eye(Ht.shape[0], dtype=torch.float64))
    return torch.linalg.svdvals(Wt).to(torch.float32).numpy()


def _collect_spectra(base_gguf: str, stats: dict | None = None, stats_g: dict | None = None):
    """Return [(out, in, singular_values)] for each factorizable dense matrix,
    plus the total original parameter count of all target tensors. When stats_g
    is given, spectra are doubly-whitened (matching the two-sided export)."""
    reader, arch = ggufio.open_reader(base_gguf)
    n_head, n_kv = ggufio.head_meta(reader, arch)
    import gguf
    spectra, orig = [], 0
    for t in reader.tensors:
        if ggufio.target_kind(t.name) != "dense" or t.tensor_type != gguf.GGMLQuantizationType.F32:
            continue
        W = np.asarray(t.data)
        orig += W.size
        base = t.name[: -len(".weight")]
        H = stats.get(base) if stats else None
        G = stats_g.get(base) if stats_g else None
        if G is not None:
            G = ggufio.permute_out_cov(G, t.name, n_head, n_kv)
        spectra.append((W.shape[0], W.shape[1], _svdvals(W, H, G)))
    return spectra, orig


def _params_at(spectra, orig, tau: float) -> float:
    """Parameter ratio if every matrix keeps energy fraction τ (break-even guarded)."""
    total = 0
    for out, in_, s in spectra:
        energy = np.cumsum(s.astype(np.float64) ** 2)
        r = int(np.searchsorted(energy / energy[-1], tau) + 1)
        r = max(1, min(r, min(out, in_)))
        r = min(-(-r // 8) * 8, min(out, in_))     # match RankPolicy.align=8
        total += (out * r + r + r * in_) if saves_params(r, out, in_) else out * in_
    return total / max(orig, 1)


def energy_for_budget(base_gguf: str, target_ratio: float, stats: dict | None = None,
                      stats_g: dict | None = None, tol: float = 0.01) -> float:
    """Binary-search the energy threshold τ whose param ratio ≈ target_ratio.
    Pass the same covariances (stats=H, stats_g=G) used for export so the
    threshold matches the (possibly doubly-)whitened spectrum it will truncate."""
    spectra, orig = _collect_spectra(base_gguf, stats, stats_g)
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) / 2
        ratio = _params_at(spectra, orig, mid)
        if abs(ratio - target_ratio) < tol:
            return mid
        if ratio < target_ratio:   # too aggressive -> raise τ
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
