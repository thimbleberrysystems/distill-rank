"""Zero-data analytic calibration: input covariances from the weights alone.

Phase 3 experiment. The data path (calibrate.py) measures H = Σ x·xᵀ per linear
by forwarding calibration text. Here we produce the same {gguf_name: H} dict
with ZERO calibration data, three ways:

  random_tokens  sample token ids from a prior, forward them (bridge baseline —
                 "propagation with no Gaussian approximation")
  mc             per-layer Gaussian moment-matched simulation: sample activations
                 from the analytically-propagated N(μ_l, Σ_l), push through the
                 REAL decoder layer (real RoPE/GQA/softmax/SiLU), re-fit moments,
                 repeat. Randomness comes only from the analytic Gaussian.
  strict         fully closed-form propagation (no sampling) — ablation variant.

Downstream (whiten_svd, planner, export) is scale-invariant in H, so per-token
moments work directly; only hybrid mixing needs trace matching (mix_stats).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .calibrate import attach_cov_hooks, gguf_name


# --- token priors -------------------------------------------------------------

def token_prior(kind: str, model_dir: str | None = None, vocab_size: int = 0, *,
                zipf_s: float = 1.0) -> np.ndarray:
    """Zero-data token frequency prior p[V].

    uniform     p ∝ 1
    zipf        p ∝ 1/(id+β)^s — BPE ids are roughly frequency-ordered
    merge_rank  BPE merge order from tokenizer.json: merges are emitted in
                training-corpus frequency order, so a token minted at merge
                rank r gets p ∝ 1/(r+r0); base (byte/char) tokens get rank 0.
    """
    if kind == "uniform":
        p = np.ones(vocab_size)
    elif kind == "zipf":
        p = 1.0 / (np.arange(vocab_size, dtype=np.float64) + 10.0) ** zipf_s
    elif kind == "merge_rank":
        tj = json.loads((Path(model_dir) / "tokenizer.json").read_text())
        vocab: dict[str, int] = tj["model"]["vocab"]
        merges = tj["model"]["merges"]
        rank = np.zeros(vocab_size, dtype=np.float64)      # base tokens: rank 0
        r0 = 256.0
        for i, m in enumerate(merges):
            a, b = m if isinstance(m, (list, tuple)) else m.split(" ", 1)
            tid = vocab.get(a + b)
            if tid is not None and tid < vocab_size:
                rank[tid] = i + 1
        # tokens outside vocab-map coverage (added/special) sit at the tail
        p = 1.0 / (rank + r0) ** zipf_s
    else:
        raise ValueError(f"unknown prior: {kind}")
    return (p / p.sum()).astype(np.float64)


# --- exact zero-propagation moments -------------------------------------------

def embedding_moments(E: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(μ0, M0) of the residual stream at layer 0 under token prior p (exact)."""
    Ew = E.astype(np.float64)
    mu = Ew.T @ p
    M = (Ew * p[:, None]).T @ Ew
    return mu, M


def layer0_exact_h(model, p: np.ndarray) -> np.ndarray:
    """H of blk.0 attn_q/k/v input — RMSNorm of individual embedding rows, so it
    is exact under the prior with no propagation or Gaussian assumption."""
    ln = model.model.layers[0].input_layernorm
    E = model.model.embed_tokens.weight.detach().to(torch.float64)
    rms = (E.pow(2).mean(-1, keepdim=True) + ln.variance_epsilon).sqrt()
    N = (E / rms) * ln.weight.detach().to(torch.float64)
    pw = torch.tensor(p, dtype=torch.float64)
    return ((N * pw[:, None]).T @ N).numpy().astype(np.float32)


# --- analytic covariance (main entry) ------------------------------------------

def _damped_chol(M: np.ndarray, mu: np.ndarray) -> np.ndarray:
    C = M - np.outer(mu, mu)
    C = (C + C.T) / 2
    eps = 1e-3 * max(np.trace(C) / C.shape[0], 1e-8)
    return np.linalg.cholesky(C + eps * np.eye(C.shape[0]))


def _gauss_batch(mu: np.ndarray, M: np.ndarray, n_seqs: int, seqlen: int,
                 rng: np.random.Generator, rho: float = 0.0) -> torch.Tensor:
    """[n_seqs, seqlen, d] samples from N(μ, Σ); optional AR(1) correlation ρ
    along the token axis (zero-data knob for attention realism)."""
    L = _damped_chol(M, mu)
    z = rng.standard_normal((n_seqs, seqlen, mu.shape[0]))
    if rho > 0.0:
        for t in range(1, seqlen):
            z[:, t] = rho * z[:, t - 1] + np.sqrt(1 - rho * rho) * z[:, t]
    x = mu + z @ L.T
    return torch.tensor(x, dtype=torch.float32)


def analytic_covariance(model_dir: str, *, prior: str = "merge_rank",
                        zipf_s: float = 1.0, mode: str = "mc",
                        samples: int = 16384, seqlen: int = 256,
                        rho: float = 0.0, seed: int = 0,
                        device: str = "cpu") -> dict[str, np.ndarray]:
    """Return {gguf_name: H float32 [in,in]} computed with zero calibration data."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device).eval()
    V = model.model.embed_tokens.weight.shape[0]
    p = token_prior(prior, model_dir, V, zipf_s=zipf_s)
    rng = np.random.default_rng(seed)
    n_seqs = max(1, samples // seqlen)

    if mode == "random_tokens":
        from .calibrate import collect_covariance
        ids = torch.tensor(rng.choice(V, size=(n_seqs, seqlen), p=p), dtype=torch.long)
        return collect_covariance(model_dir, [], None, device=device, token_ids=ids)

    if mode not in ("mc", "noise"):
        raise ValueError(f"unknown mode: {mode} (mc | noise | random_tokens)")

    H, handles = attach_cov_hooks(model)
    pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    causal = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=device),
                        diagonal=1)[None, None]

    E = model.model.embed_tokens.weight.detach().cpu().numpy()
    if mode == "noise":
        # isotropic Gaussian noise at the embedding RMS, propagated through the
        # real layers. Control experiment: whitening needs covariance directions,
        # which noise lacks — this scores WORSE than plain SVD (see README).
        d = E.shape[1]
        mu, M = np.zeros(d), float(E.std()) ** 2 * np.eye(d)
    else:
        mu, M = embedding_moments(E, p)
    with torch.no_grad():
        for layer in model.model.layers:
            X = _gauss_batch(mu, M, n_seqs, seqlen, rng, rho).to(device)
            pe = model.model.rotary_emb(X, pos_ids)
            outs = []
            for i in range(n_seqs):   # per-seq to bound memory; hooks accumulate
                out = layer(X[i:i + 1], attention_mask=causal,
                            position_ids=pos_ids, position_embeddings=pe)
                outs.append((out[0] if isinstance(out, tuple) else out).squeeze(0))
            Y = torch.cat(outs).to(torch.float64).numpy()   # [n_seqs*seqlen, d]
            mu, M = Y.mean(0), (Y.T @ Y) / Y.shape[0]       # re-Gaussianize
        # final-norm output feeds the (possibly tied) lm_head -> 'output'
        Xf = _gauss_batch(mu, M, n_seqs, seqlen, rng, rho).to(device)
        normed = model.model.norm(Xf).reshape(-1, Xf.shape[-1]).to(torch.float64)
        H["output"] = (normed.T @ normed).to(torch.float32)

    for h in handles:
        h.remove()
    stats = {g: v.cpu().numpy().astype(np.float32) for g, v in H.items()}
    if mode == "mc":
        # blk.0 attn input is exact under the prior — no reason to keep the MC
        # estimate. (Skipped for noise: it has no prior / embedding manifold.)
        exact0 = layer0_exact_h(model, p)
        for k in ("attn_q", "attn_k", "attn_v"):
            key = f"blk.0.{k}"
            if key in stats:
                stats[key] = exact0
    return stats


# --- zero-data two-sided (Fisher) calibration -----------------------------------

def collect_influence_prior(model_dir: str, *, prior: str = "merge_rank",
                            zipf_s: float = 1.0, samples: int = 8192, seqlen: int = 256,
                            seed: int = 0, device: str = "cpu") -> tuple[dict, dict]:
    """Zero-data two-sided calibration: sample token ids from the merge-rank
    prior, run forward+backward under the LM loss, and collect BOTH the input
    covariance H = Σ xxᵀ and the output-gradient covariance G = Σ ggᵀ.

    This is the data-free analog of calibrate.collect_influence — the output-side
    Fisher signal for IO-SVD-style two-sided whitening, obtained with no
    calibration text (novel combination of IO-SVD with the Phase-3 zero-data
    prior). Returns ({gguf_name: H}, {gguf_name: G}).
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device).eval()
    V = model.model.embed_tokens.weight.shape[0]
    p = token_prior(prior, model_dir, V, zipf_s=zipf_s)
    rng = np.random.default_rng(seed)
    n = max(1, samples // seqlen)
    ids = torch.tensor(rng.choice(V, size=(n, seqlen), p=p), dtype=torch.long)

    H, fwd = attach_cov_hooks(model)
    G: dict = {}
    bwd = []

    def make_bhook(gname):
        def hook(_mod, _gi, grad_out):
            g = grad_out[0].detach().reshape(-1, grad_out[0].shape[-1]).to(torch.float32)
            gtg = g.t() @ g
            G[gname] = gtg if gname not in G else G[gname] + gtg
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            gn = gguf_name(name)
            if gn is not None and gn != "output":   # lm_head G is [vocab,vocab]: skip
                bwd.append(mod.register_full_backward_hook(make_bhook(gn)))

    for i in range(n):
        chunk = ids[i:i + 1].to(device)
        model.zero_grad(set_to_none=True)
        model(chunk, labels=chunk).loss.backward()

    for h in fwd + bwd:
        h.remove()
    Hn = {g: v.cpu().numpy().astype(np.float32) for g, v in H.items()}
    Gn = {g: v.cpu().numpy().astype(np.float32) for g, v in G.items()}
    return Hn, Gn


# --- hybrid mixing --------------------------------------------------------------

def mix_stats(h_analytic: dict, h_data: dict, lam: float, *,
              match: str = "trace") -> dict:
    """H = λ·Ĥ_analytic(rescaled) + (1−λ)·H_data, per key. Keys only in one dict
    pass through unchanged."""
    out = {}
    for k in set(h_analytic) | set(h_data):
        a, d = h_analytic.get(k), h_data.get(k)
        if a is None or d is None:
            out[k] = d if a is None else a
            continue
        if match == "trace":
            ta, td = float(np.trace(a)), float(np.trace(d))
            a = a * (td / ta) if ta > 0 else a
        out[k] = (lam * a + (1.0 - lam) * d).astype(np.float32)
    return out


# --- diagnostics -----------------------------------------------------------------

def _eig_desc(H: np.ndarray):
    w, v = np.linalg.eigh(H.astype(np.float64))
    return np.maximum(w[::-1], 0.0), v[:, ::-1]


def compare_stats(h_a: dict, h_b: dict, *, topk: int = 32) -> dict[str, dict]:
    """Per-key agreement of analytic (a) vs reference (b) covariances:
    relF            ‖A−B‖_F/‖B‖_F after trace-normalizing both
    overlap         plain top-k subspace overlap ‖UaᵀUb‖²_F/k
    energy_capture  fraction of B's total energy captured by A's top-k subspace
                    (eigenvalue-weighted — the metric that matters for whitening,
                    since measured spectra are extremely top-heavy)
    """
    out = {}
    for k in sorted(set(h_a) & set(h_b)):
        A = h_a[k].astype(np.float64); A /= max(np.trace(A), 1e-30)
        B = h_b[k].astype(np.float64); B /= max(np.trace(B), 1e-30)
        wa, va = _eig_desc(A)
        wb, vb = _eig_desc(B)
        Ua, Ub = va[:, :topk], vb[:, :topk]
        proj = Ua.T @ vb                       # [topk, d] coords of B's eigvecs in A-span
        out[k] = {
            "relF": float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-30)),
            "overlap": float((np.linalg.norm(Ua.T @ Ub) ** 2) / topk),
            "energy_capture": float((wb * (proj ** 2).sum(0)).sum() / max(wb.sum(), 1e-30)),
        }
    return out


def whitened_agreement(base_gguf: str, h_a: dict, h_b: dict, tau: float = 0.9) -> dict:
    """For each dense tensor in the GGUF: whitened spectra under both H's and the
    induced energy-rank at τ — the quantity the planner/export actually consume."""
    from . import ggufio
    from .planner import _svdvals
    import gguf as gguf_mod

    reader, _ = ggufio.open_reader(base_gguf)
    out = {}
    for t in reader.tensors:
        if ggufio.target_kind(t.name) != "dense" or t.tensor_type != gguf_mod.GGMLQuantizationType.F32:
            continue
        base = t.name[: -len(".weight")]
        if base not in h_a or base not in h_b:
            continue
        W = np.asarray(t.data)
        ranks = []
        for H in (h_a[base], h_b[base]):
            s = _svdvals(W, H).astype(np.float64)
            energy = np.cumsum(s ** 2)
            ranks.append(int(np.searchsorted(energy / energy[-1], tau) + 1))
        ra, rb = ranks
        out[base] = {"rank_a": ra, "rank_b": rb,
                     "rank_rel_diff": abs(ra - rb) / max(rb, 1)}
    return out
