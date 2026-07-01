#!/usr/bin/env python3
"""distill-rank - SVD-factorize the linear weights of an existing GGUF.

Reads a plain (f32/f16) GGUF and writes a new GGUF in which every targeted
linear weight W (shape [out, in], or [n_expert, out, in] for MoE expert stacks)
is replaced by three tensors from its thin SVD:

    W = U @ diag(s) @ Vt        U:[out, r]   s:[r]   Vt:[r, in]   r = min(out, in)

At full rank this is loss-less; `--rank N` truncates to a genuine low-rank
approximation. Working on the *final* GGUF (rather than hooking the HF
converter) guarantees the factors correspond exactly to the weights the runtime
loads, and is architecture-agnostic: it keys off the canonical GGUF tensor names
shared by every architecture. The patched llama.cpp consumes the factors through
the universal build_lora_mm / build_lora_mm_id ops.

Usage:
    python svd_export.py in.gguf out.gguf [--rank N]
Produce the input GGUF first, e.g.:
    python vendor/llama.cpp/convert_hf_to_gguf.py models/X --outfile in.gguf --outtype f32
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", "vendor/llama.cpp")).resolve()
sys.path.insert(0, str(LLAMA_CPP_DIR / "gguf-py"))

import numpy as np  # noqa: E402
import gguf  # noqa: E402

# Canonical per-layer linear "kinds" (between `blk.N.` and `.weight`) that every
# architecture applies through build_lora_mm / _id. Excludes embeddings, norms,
# biases, and exotic direct-matmul weights (MLA attn_*_a/_b, SSM/conv).
_DENSE = {
    "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
}
_MOE = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"}
_BLK = re.compile(r"^blk\.\d+\.([a-z0-9_]+)\.weight$")


def _kind(name: str) -> str | None:
    if name == "output.weight":
        return "dense"
    m = _BLK.match(name)
    if not m:
        return None
    k = m.group(1)
    return "dense" if k in _DENSE else "moe" if k in _MOE else None


def _svd2d(W: np.ndarray, rank: int | None):
    """W[out,in] -> (U[out,r], s[r], Vt[r,in]); returns (factors, max_abs_err)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    r = s.shape[0] if rank is None else min(rank, s.shape[0])
    U, s, Vt = U[:, :r], s[:r], Vt[:r, :]
    err = float(np.abs((U * s) @ Vt - W).max())
    return (np.ascontiguousarray(U), np.ascontiguousarray(s),
            np.ascontiguousarray(Vt)), err


def _copy_kv(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip or not field.types:
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_array(key, field.contents())
        else:
            writer.add_key_value(key, field.contents(), vtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--rank", type=int, default=None, help="cap rank (default: full = loss-less)")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.infile)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(args.outfile, arch)
    _copy_kv(reader, writer)

    n_fact = 0
    max_err = 0.0
    for t in reader.tensors:
        kind = _kind(t.name)
        is_f32 = t.tensor_type == gguf.GGMLQuantizationType.F32
        data = np.asarray(t.data)  # logical shape [out, in] (or [n_expert, out, in])
        if kind is None or not is_f32:
            writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
            continue

        base = t.name[: -len(".weight")]
        if kind == "dense":
            assert data.ndim == 2, f"{t.name}: expected 2D, got {data.shape}"
            (U, s, Vt), err = _svd2d(data, args.rank)
        else:  # moe: [n_expert, out, in] -> per-expert SVD, stacked
            assert data.ndim == 3, f"{t.name}: expected 3D MoE, got {data.shape}"
            facs = [_svd2d(data[e], args.rank) for e in range(data.shape[0])]
            U = np.stack([f[0][0] for f in facs])
            s = np.stack([f[0][1] for f in facs])
            Vt = np.stack([f[0][2] for f in facs])
            err = max(f[1] for f in facs)
        max_err = max(max_err, err)
        n_fact += 1
        writer.add_tensor(base + ".svd_u", U, raw_dtype=gguf.GGMLQuantizationType.F32)
        writer.add_tensor(base + ".svd_s", s, raw_dtype=gguf.GGMLQuantizationType.F32)
        writer.add_tensor(base + ".svd_vt", Vt, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    tag = "full rank" if args.rank is None else f"rank<={args.rank}"
    print(f"[svd_export] arch={arch} factorized {n_fact} weights ({tag}); "
          f"max reconstruction error {max_err:g}", file=sys.stderr)


if __name__ == "__main__":
    main()
