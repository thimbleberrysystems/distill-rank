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

    kind: 'full' | 'fixed' | 'frac' | 'energy' | 'error'
      full   -> r = min(out,in)                       (loss-less)
      fixed  -> r = value                             (clamped to min(out,in))
      frac   -> r = round(value * min(out,in))
      energy -> smallest r with sum(top-r s^2)/sum(s^2) >= value
      error  -> quality-driven dynamic rank: smallest r whose *relative error in
                the metric of s* is <= value (=ε). On the activation-aware path s
                is the whitened spectrum svdvals(W·S), and
                sum_{i>r} s_i^2 / sum s_i^2 == ‖(W−W_r)X‖²/‖WX‖², so ε is the
                per-matrix activation error. Identical mechanics to `energy` with
                threshold (1−ε), but parameterized by a *quality* knob rather than
                an energy fraction — each matrix keeps whatever rank meets ε, so
                the model size emerges instead of being imposed.

    align: kept ranks are rounded UP to this multiple (then clamped to full).
      ggml's fast f32 GEMM (llamafile sgemm) bails out when the inner dimension
      k % 8 != 0 — and in the factored path the second matmul's inner dim IS the
      rank — so an unaligned rank silently drops every U-side matmul onto the
      slow generic path. Rounding up costs <2% params and keeps strictly more
      spectrum, so it can only help quality.
    """
    kind: str = "full"
    value: float = 1.0
    align: int = 8

    def choose(self, s: np.ndarray, out: int, in_: int) -> int:
        k = min(out, in_)
        if self.kind == "full":
            return k
        if self.kind == "fixed":
            r = max(1, min(int(self.value), k))
        elif self.kind == "frac":
            r = max(1, min(int(round(self.value * k)), k))
        elif self.kind in ("energy", "error"):
            thresh = self.value if self.kind == "energy" else 1.0 - self.value
            energy = np.cumsum(s.astype(np.float64) ** 2)
            total = energy[-1] if energy[-1] > 0 else 1.0
            r = int(np.searchsorted(energy / total, thresh) + 1)
            r = max(1, min(r, k))
        else:
            raise ValueError(f"unknown rank policy: {self.kind}")
        if self.align > 1:
            r = min(-(-r // self.align) * self.align, k)
        return r


def saves_params(r: int, out: int, in_: int) -> bool:
    """True if the factored form is smaller than the dense matrix."""
    return r * (out + in_) < out * in_


# --- SVD ---------------------------------------------------------------------

def whiten_svd(W: np.ndarray, H: np.ndarray, policy: "RankPolicy",
               damp: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Activation-aware truncation (SVD-LLM style).

    Given input covariance H (=Σ x xᵀ), truncate W in the H-metric: factor W·S
    (H = S Sᵀ), keep rank r, then de-whiten. Returns runtime factors
    (U[out,r], s[r], Vt[r,in]) with W_r = U diag(s) Vt ≈ W, minimizing ‖(W-W_r)·X‖.
    """
    Wt = torch.tensor(np.ascontiguousarray(W), dtype=torch.float64)
    Ht = torch.tensor(np.ascontiguousarray(H), dtype=torch.float64)
    n = Ht.shape[0]
    eps = damp * torch.diagonal(Ht).mean().clamp(min=1e-8)
    Ht = Ht + eps * torch.eye(n, dtype=torch.float64)
    S = torch.linalg.cholesky(Ht)                       # H = S Sᵀ (S lower-triangular)
    U, sig, Vt = torch.linalg.svd(Wt @ S, full_matrices=False)
    r = policy.choose(sig.numpy().astype(np.float32), W.shape[0], W.shape[1])
    U, sig, Vt = U[:, :r], sig[:r], Vt[:r, :]
    # svd_vt = Vt @ S^{-1}  ->  svd_vtᵀ = (Sᵀ)^{-1} Vtᵀ = solve_triangular(Sᵀ, Vtᵀ, upper)
    svd_vt = torch.linalg.solve_triangular(
        S.transpose(-1, -2), Vt.transpose(-1, -2), upper=True).transpose(-1, -2)
    err = float(((U * sig) @ svd_vt - Wt).abs().max())
    return (np.ascontiguousarray(U.to(torch.float32).numpy()),
            np.ascontiguousarray(sig.to(torch.float32).numpy()),
            np.ascontiguousarray(svd_vt.to(torch.float32).numpy()), err)


def _damped_cholesky(M: np.ndarray, damp: float = 1e-3):
    """chol(M + damp·mean(diag)·I) as float64 torch — reusable across calls."""
    Mt = torch.tensor(np.ascontiguousarray(M), dtype=torch.float64)
    eps = damp * torch.diagonal(Mt).mean().clamp(min=1e-8)
    return torch.linalg.cholesky(Mt + eps * torch.eye(Mt.shape[0], dtype=torch.float64))


def two_sided_whiten_svd(W: np.ndarray, H: np.ndarray, G: np.ndarray, policy: "RankPolicy",
                         damp: float = 1e-3, S=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Input-output (Fisher-weighted) truncation, cf. IO-SVD / GFWSVD.

    H = Σ xxᵀ (input activations), G = Σ ggᵀ (output loss-gradients). Factor
    the doubly-whitened matrix L·W·S (H=SSᵀ, G=LLᵀ), keep rank r, then un-whiten
    both sides. Minimizes the 2nd-order loss impact tr((W−W_r)ᵀG(W−W_r)H) instead
    of plain activation error ‖(W−W_r)X‖. Returns runtime factors (U,s,Vt).
    Pass a pre-computed S = chol(H+damp) to avoid recomputing it for rank selection.
    """
    Wt = torch.tensor(np.ascontiguousarray(W), dtype=torch.float64)
    if S is None:
        S = _damped_cholesky(H, damp)                                             # H = S Sᵀ
    L = _damped_cholesky(G, damp)                                                  # G = L Lᵀ
    U, sig, Vt = torch.linalg.svd(L @ Wt @ S, full_matrices=False)                 # doubly whitened
    r = policy.choose(sig.numpy().astype(np.float32), W.shape[0], W.shape[1])
    U, sig, Vt = U[:, :r], sig[:r], Vt[:r, :]
    # un-whiten: svd_u = L^{-ᵀ}... W_r = L^{-1}(U diag(sig)) (Vt S^{-1}); fold sig into U side
    svd_u = torch.linalg.solve_triangular(L, U * sig, upper=False)                 # L svd_u = U·sig
    svd_vt = torch.linalg.solve_triangular(
        S.transpose(-1, -2), Vt.transpose(-1, -2), upper=True).transpose(-1, -2)   # Vt S^{-1}
    ones = torch.ones(r, dtype=torch.float64)
    err = float(((svd_u) @ svd_vt - Wt).abs().max())
    return (np.ascontiguousarray(svd_u.to(torch.float32).numpy()),
            np.ascontiguousarray(ones.to(torch.float32).numpy()),
            np.ascontiguousarray(svd_vt.to(torch.float32).numpy()), err)


def svd_truncate(W: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (U[out,r], s[r], Vt[r,in], max_abs_reconstruction_error)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    U, s, Vt = _svd(W)
    U, s, Vt = U[:, :r], s[:r], Vt[:r, :]
    err = float(np.abs((U * s) @ Vt - W).max())
    return (np.ascontiguousarray(U), np.ascontiguousarray(s),
            np.ascontiguousarray(Vt), err)
