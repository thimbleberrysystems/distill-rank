"""Factorization math + rank-selection policies (data-free, numpy).

A weight W is [out, in]. Thin SVD: W = U diag(s) Vt, U:[out,r] s:[r] Vt:[r,in].
The patched runtime stores/consumes exactly (svd_u=U, svd_s=s, svd_vt=Vt).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# numpy's LAPACK on some boxes is the slow reference build (~1s per small SVD);
# torch's is ~10x faster, so use it for the decomposition when available.
try:
    import torch

    def _svd(Wc: np.ndarray):
        # torch.tensor copies (Wc may be a read-only mmap view); cost is negligible vs SVD
        U, s, Vt = torch.linalg.svd(torch.tensor(Wc, dtype=torch.float32), full_matrices=False)
        return U.numpy(), s.numpy(), Vt.numpy()
except ImportError:  # pragma: no cover
    def _svd(Wc: np.ndarray):
        return np.linalg.svd(Wc, full_matrices=False)


# --- rank policies -----------------------------------------------------------

@dataclass
class RankPolicy:
    """How to pick the kept rank r for a matrix.

    kind: 'full' | 'fixed' | 'frac' | 'energy'
      full   -> r = min(out,in)                       (loss-less)
      fixed  -> r = value                             (clamped to min(out,in))
      frac   -> r = round(value * min(out,in))
      energy -> smallest r with sum(top-r s^2)/sum(s^2) >= value
    """
    kind: str = "full"
    value: float = 1.0

    def choose(self, s: np.ndarray, out: int, in_: int) -> int:
        k = min(out, in_)
        if self.kind == "full":
            return k
        if self.kind == "fixed":
            return max(1, min(int(self.value), k))
        if self.kind == "frac":
            return max(1, min(int(round(self.value * k)), k))
        if self.kind == "energy":
            energy = np.cumsum(s.astype(np.float64) ** 2)
            total = energy[-1] if energy[-1] > 0 else 1.0
            r = int(np.searchsorted(energy / total, self.value) + 1)
            return max(1, min(r, k))
        raise ValueError(f"unknown rank policy: {self.kind}")


def saves_params(r: int, out: int, in_: int) -> bool:
    """True if the factored form is smaller than the dense matrix."""
    return r * (out + in_) < out * in_


# --- SVD ---------------------------------------------------------------------

def svd_truncate(W: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (U[out,r], s[r], Vt[r,in], max_abs_reconstruction_error)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    U, s, Vt = _svd(W)
    U, s, Vt = U[:, :r], s[:r], Vt[:r, :]
    err = float(np.abs((U * s) @ Vt - W).max())
    return (np.ascontiguousarray(U), np.ascontiguousarray(s),
            np.ascontiguousarray(Vt), err)
