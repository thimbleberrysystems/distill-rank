"""Collect per-linear input-activation covariance for activation-aware SVD.

For each target linear with input x, we accumulate H = Σ_tokens x xᵀ  (shape [in,in])
over a calibration set. Activation-aware factorization then truncates W in the
metric induced by H, i.e. minimizes ‖(W - W_r)·X‖ rather than ‖W - W_r‖ — which
is what actually matters for the model's outputs.

Stats are keyed by the GGUF tensor name (blk.N.attn_q, ffn_down, output, ...) so
they plug straight into the GGUF-native export.
"""
from __future__ import annotations

import re

import numpy as np
import torch

# HF module-name suffix -> GGUF tensor base. Covers the llama/qwen/gemma family
# (separate q/k/v, gate/up) and phi3-style fused projections (qkv_proj, gate_up_proj).
# Order matters: check the fused names before the shorter suffixes they contain.
_HF2GGUF = {
    "qkv_proj": "attn_qkv",       # phi3 fused QKV
    "gate_up_proj": "ffn_up",     # phi3 fused gate+up (GGUF stores as ffn_up)
    "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
    "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down",
}
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def gguf_name(module_name: str) -> str | None:
    """Map an HF Linear module path to its GGUF tensor name, or None if not a target."""
    if module_name.endswith("lm_head"):
        return "output"
    for suf, base in _HF2GGUF.items():
        if module_name.endswith(suf):
            m = _LAYER.search(module_name)
            if not m:
                return None
            return f"blk.{m.group(1)}.{base}"
    return None


def _wants_output_cov(gname: str) -> bool:
    """Which linears get an output-side covariance G for two-sided whitening.

    Only matrices whose output feeds ~linearly into the residual stream, where
    the Gauss-Newton/Fisher approximation behind IO-SVD holds: attn_v, attn_output,
    ffn_gate/up/down. Excluded: attn_q/k/qkv (their outputs go through the softmax
    attention scores — a strongly nonlinear path where two-sided *hurts*, measured
    ~2.7× worse PPL) and output/lm_head (its G is [vocab,vocab] — intractable and
    rank-deficient)."""
    return not (gname.endswith("attn_q") or gname.endswith("attn_k")
                or gname.endswith("attn_qkv") or gname == "output")


def attach_cov_hooks(model) -> tuple[dict, list]:
    """Register H += xᵀx forward hooks on every target Linear.

    Returns (accumulator {gguf_name: torch [in,in]}, hook handles). Shared by the
    data path below and the analytic/random-token paths in analytic.py so all
    calibration sources accumulate identically.
    """
    H: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(gname):
        def hook(_mod, inp, _out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)
            acc = H.get(gname)
            xtx = x.t() @ x
            H[gname] = xtx if acc is None else acc + xtx
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            g = gguf_name(name)
            if g is not None:
                handles.append(mod.register_forward_hook(make_hook(g)))
    return H, handles


def collect_influence(model_dir: str, texts: list[str], tokenizer, *,
                      seqlen: int = 512, max_seqs: int = 32, device: str = "auto") -> tuple[dict, dict]:
    """Return ({gguf_name: H [in,in]}, {gguf_name: G [out,out]}) for two-sided
    (Fisher-weighted) whitening: H = Σ xxᵀ over inputs (activation statistics),
    G = Σ ggᵀ over output gradients g = ∂loss/∂y (output loss-sensitivity).

    Truncating the SVD of L·W·S (H=SSᵀ, G=LLᵀ) minimizes the 2nd-order loss
    impact tr((W−W_r)ᵀ G (W−W_r) H) rather than plain activation error — the
    input-output / Fisher-weighted objective (cf. IO-SVD arXiv:2605.15626,
    GFWSVD arXiv:2505.17974). Uses the LM loss as the differentiated signal.
    """
    from transformers import AutoModelForCausalLM

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device).eval()

    H, fwd = attach_cov_hooks(model)
    G: dict[str, torch.Tensor] = {}
    bwd = []

    def make_bhook(gname):
        def hook(_mod, grad_in, grad_out):
            g = grad_out[0].detach().reshape(-1, grad_out[0].shape[-1]).to(torch.float32)
            gtg = g.t() @ g
            G[gname] = gtg if gname not in G else G[gname] + gtg
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            gname = gguf_name(name)
            if gname is not None and _wants_output_cov(gname):
                bwd.append(mod.register_full_backward_hook(make_bhook(gname)))

    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n = min(max_seqs, ids.shape[0] // seqlen)
    for i in range(n):
        chunk = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        out = model(chunk, labels=chunk)
        out.loss.backward()

    for h in fwd + bwd:
        h.remove()
    Hn = {g: v.cpu().numpy().astype(np.float32) for g, v in H.items()}
    Gn = {g: v.cpu().numpy().astype(np.float32) for g, v in G.items()}
    return Hn, Gn


def collect_covariance(model_dir: str, texts: list[str], tokenizer, *,
                       seqlen: int = 512, max_seqs: int = 128, device: str = "auto",
                       token_ids: "torch.Tensor | None" = None) -> dict:
    """Return {gguf_name: H (float32 [in,in] numpy)} accumulated over the calibration set.

    token_ids overrides the text pipeline with a pre-built [n_seqs, seqlen] id
    tensor (used by the zero-data random-token calibration source).
    """
    from transformers import AutoModelForCausalLM

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device).eval()

    H, handles = attach_cov_hooks(model)

    if token_ids is None:
        # build fixed-length token windows from the calibration text
        ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
        n = min(max_seqs, ids.shape[0] // seqlen)
        token_ids = torch.stack([ids[i * seqlen:(i + 1) * seqlen] for i in range(n)])
    with torch.no_grad():
        for chunk in token_ids:
            model(chunk.unsqueeze(0).to(device))

    for h in handles:
        h.remove()
    return {g: v.cpu().numpy().astype(np.float32) for g, v in H.items()}
