"""GGUF read/write helpers shared by the factorization + export code.

Keeps the target-tensor naming in one place: the canonical per-layer linear
"kinds" that the patched runtime applies through build_lora_mm / build_lora_mm_id.
Excludes embeddings, norms, biases and exotic direct-matmul weights.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", "vendor/llama.cpp")).resolve()
_gguf_py = LLAMA_CPP_DIR / "gguf-py"
if _gguf_py.is_dir():
    sys.path.insert(0, str(_gguf_py))

import gguf  # noqa: E402

_DENSE = {
    "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
}
_MOE = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"}
_BLK = re.compile(r"^blk\.\d+\.([a-z0-9_]+)\.weight$")


def target_kind(name: str) -> str | None:
    """Return 'dense', 'moe', or None for a GGUF tensor name."""
    if name == "output.weight":
        return "dense"
    m = _BLK.match(name)
    if not m:
        return None
    k = m.group(1)
    return "dense" if k in _DENSE else "moe" if k in _MOE else None


def copy_kv(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    """Copy all key/value metadata except the fields the writer manages itself."""
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip or not field.types:
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_array(key, field.contents())
        else:
            writer.add_key_value(key, field.contents(), vtype)


def open_reader(path: str) -> tuple[gguf.GGUFReader, str]:
    reader = gguf.GGUFReader(path)
    arch = reader.fields["general.architecture"].contents()
    return reader, arch


def head_meta(reader, arch: str) -> tuple[int | None, int | None]:
    """(n_head, n_head_kv) from GGUF attention metadata, or (None, None)."""
    import numpy as _np  # noqa

    def get(k):
        f = reader.fields.get(f"{arch}.attention.{k}")
        return int(f.contents()) if f else None
    nh = get("head_count")
    return nh, (get("head_count_kv") or nh)


def rope_perm(out: int, n_head: int):
    """Row permutation convert_hf_to_gguf applies to q/k: gguf_row = hf_row[perm]."""
    import numpy as np
    return np.arange(out).reshape(n_head, 2, out // n_head // 2).swapaxes(1, 2).reshape(-1)


def permute_out_cov(G, name: str, n_head: int | None, n_kv: int | None):
    """Permute an output-side covariance (HF layout) into GGUF layout for q/k,
    whose output rows are RoPE-permuted by the converter; identity otherwise."""
    import numpy as np
    if name.endswith("attn_q.weight") and n_head:
        p = rope_perm(G.shape[0], n_head)
    elif name.endswith("attn_k.weight") and n_kv:
        p = rope_perm(G.shape[0], n_kv)
    else:
        return G
    return np.ascontiguousarray(G[np.ix_(p, p)])
