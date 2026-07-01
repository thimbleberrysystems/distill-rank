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


def _svdvals(W: np.ndarray, H: np.ndarray | None) -> np.ndarray:
    """Singular values used for the energy metric — whitened by H (=svdvals(W·S),
    H=SSᵀ) when covariance is supplied, so the threshold matches the
    activation-aware export; otherwise the plain singular values of W."""
    Wt = torch.tensor(np.ascontiguousarray(W), dtype=torch.float64)
    if H is not None:
        Ht = torch.tensor(np.ascontiguousarray(H), dtype=torch.float64)
        eps = 1e-3 * torch.diagonal(Ht).mean().clamp(min=1e-8)
        Wt = Wt @ torch.linalg.cholesky(Ht + eps * torch.eye(Ht.shape[0], dtype=torch.float64))
    return torch.linalg.svdvals(Wt).to(torch.float32).numpy()


def _collect_spectra(base_gguf: str, stats: dict | None = None):
    """Return [(out, in, singular_values)] for each factorizable dense matrix,
    plus the total original parameter count of all target tensors."""
    reader, _ = ggufio.open_reader(base_gguf)
    import gguf
    spectra, orig = [], 0
    for t in reader.tensors:
        if ggufio.target_kind(t.name) != "dense" or t.tensor_type != gguf.GGMLQuantizationType.F32:
            continue
        W = np.asarray(t.data)
        orig += W.size
        H = stats.get(t.name[: -len(".weight")]) if stats else None
        spectra.append((W.shape[0], W.shape[1], _svdvals(W, H)))
    return spectra, orig


def _params_at(spectra, orig, tau: float) -> float:
    """Parameter ratio if every matrix keeps energy fraction τ (break-even guarded)."""
    total = 0
    for out, in_, s in spectra:
        energy = np.cumsum(s.astype(np.float64) ** 2)
        r = int(np.searchsorted(energy / energy[-1], tau) + 1)
        r = max(1, min(r, min(out, in_)))
        total += (out * r + r + r * in_) if saves_params(r, out, in_) else out * in_
    return total / max(orig, 1)


def energy_for_budget(base_gguf: str, target_ratio: float, stats: dict | None = None,
                      tol: float = 0.01) -> float:
    """Binary-search the energy threshold τ whose param ratio ≈ target_ratio.
    Pass the same covariance `stats` used for activation-aware export so the
    threshold matches the whitened spectrum the export will truncate."""
    spectra, orig = _collect_spectra(base_gguf, stats)
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
