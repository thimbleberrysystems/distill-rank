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

# HF module-name suffix -> GGUF tensor base (llama/qwen2 family naming)
_HF2GGUF = {
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


def collect_covariance(model_dir: str, texts: list[str], tokenizer, *,
                       seqlen: int = 512, max_seqs: int = 128, device: str = "auto") -> dict:
    """Return {gguf_name: H (float32 [in,in] numpy)} accumulated over the calibration set."""
    from transformers import AutoModelForCausalLM

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.to(device).eval()

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

    # build fixed-length token windows from the calibration text
    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n = min(max_seqs, ids.shape[0] // seqlen)
    with torch.no_grad():
        for i in range(n):
            chunk = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(device)
            model(chunk)

    for h in handles:
        h.remove()
    return {g: v.cpu().numpy().astype(np.float32) for g, v in H.items()}
