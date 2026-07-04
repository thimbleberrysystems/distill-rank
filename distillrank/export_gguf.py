"""Write a factored GGUF from a baseline GGUF, applying a rank policy per weight.

For each targeted linear:
  - pick rank r from the policy (using that matrix's singular values),
  - if the factored form is smaller (break-even guard) -> write svd_u/s/vt,
  - otherwise keep the dense weight unchanged.
MoE expert stacks ([n_expert, out, in]) are factorized per expert and re-stacked.
Non-target tensors (embeddings, norms, biases, router) are copied verbatim.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from . import ggufio
from .factorize import (RankPolicy, saves_params, svd_truncate, whiten_svd,
                        two_sided_whiten_svd, _svd)

import gguf  # noqa: E402  (path set up by ggufio)

from .ggufio import head_meta as _head_meta, permute_out_cov as _g_to_gguf  # noqa: E402


@dataclass
class ExportStats:
    factorized: int = 0
    kept_dense: int = 0
    max_err: float = 0.0
    orig_params: int = 0
    new_params: int = 0
    per_tensor: list = field(default_factory=list)  # (name, out, in, r, err)


def _factor_2d(W: np.ndarray, policy: RankPolicy, st: ExportStats, name: str, H=None, G=None):
    out, in_ = W.shape
    if H is not None and G is not None:     # two-sided (Fisher-weighted, IO-SVD)
        U, s, Vt, err = two_sided_whiten_svd(W, H, G, policy)
    elif H is not None:                     # activation-aware (input only)
        U, s, Vt, err = whiten_svd(W, H, policy)
    else:                                   # plain SVD (single, torch-backed)
        Wc = np.ascontiguousarray(W, dtype=np.float32)
        U0, s0, Vt0 = _svd(Wc)
        r = policy.choose(s0, out, in_)
        U = np.ascontiguousarray(U0[:, :r]); s = np.ascontiguousarray(s0[:r]); Vt = np.ascontiguousarray(Vt0[:r, :])
        err = float(np.abs((U * s) @ Vt - Wc).max())
    r = s.shape[0]
    if not saves_params(r, out, in_):
        return None
    st.max_err = max(st.max_err, err)
    st.per_tensor.append((name, out, in_, r, err))
    st.new_params += U.size + s.size + Vt.size
    return U, s, Vt


def export(infile: str, outfile: str, policy: RankPolicy, stats: dict | None = None,
           stats_g: dict | None = None, merge: bool = False) -> ExportStats:
    """Factorize each target linear.

    stats:   optional {gguf_base_name: input covariance H} for activation-aware truncation.
    stats_g: optional {gguf_base_name: output covariance G} — when present with H, uses
             two-sided (Fisher-weighted, IO-SVD) truncation. q/k G are permuted to GGUF layout.
    merge:   if True, write the reconstructed dense low-rank weight W' = U diag(s) Vt under
             the original tensor name (same size, runs on STOCK Ollama) instead of the
             factors — for measuring rank-r quality without the patched runtime.
    """
    reader, arch = ggufio.open_reader(infile)
    writer = gguf.GGUFWriter(outfile, arch)
    ggufio.copy_kv(reader, writer)
    n_head, n_kv = _head_meta(reader, arch)
    st = ExportStats()

    for t in reader.tensors:
        kind = ggufio.target_kind(t.name)
        data = np.asarray(t.data)
        is_f32 = t.tensor_type == gguf.GGMLQuantizationType.F32
        if kind is None or not is_f32:
            writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
            continue

        st.orig_params += data.size
        base = t.name[: -len(".weight")]

        if kind == "dense":
            H = stats.get(base) if stats else None
            G = stats_g.get(base) if stats_g else None
            if G is not None:
                G = _g_to_gguf(G, t.name, n_head, n_kv)
            res = _factor_2d(data, policy, st, t.name, H=H, G=G)
            if res is None:
                st.kept_dense += 1
                st.new_params += data.size
                writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
                continue
            U, s, Vt = res
        else:  # moe: [n_expert, out, in] -> per-expert
            per = [_factor_2d(data[e], policy, st, f"{t.name}[{e}]") for e in range(data.shape[0])]
            if any(p is None for p in per):
                st.kept_dense += 1
                st.new_params += data.size
                writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
                continue
            U = np.stack([p[0] for p in per])
            s = np.stack([p[1] for p in per])
            Vt = np.stack([p[2] for p in per])

        st.factorized += 1
        if merge:  # reconstruct dense W' = U diag(s) Vt (stock-Ollama compatible)
            if U.ndim == 2:
                Wr = (U * s) @ Vt
            else:  # MoE stack
                Wr = np.stack([(U[e] * s[e]) @ Vt[e] for e in range(U.shape[0])])
            writer.add_tensor(t.name, np.ascontiguousarray(Wr, dtype=np.float32),
                              raw_dtype=gguf.GGMLQuantizationType.F32)
            continue
        writer.add_tensor(base + ".svd_u", U, raw_dtype=gguf.GGMLQuantizationType.F32)
        writer.add_tensor(base + ".svd_s", s, raw_dtype=gguf.GGMLQuantizationType.F32)
        writer.add_tensor(base + ".svd_vt", Vt, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return st


def export_from_factors(infile: str, outfile: str, factors: dict) -> ExportStats:
    """Write a factored GGUF using precomputed factors (e.g. finetuned).

    factors: {gguf_base_name: (U[out,r], s[r], Vt[r,in])} for dense linears.
    Tensors not in `factors` are copied verbatim.
    """
    reader, arch = ggufio.open_reader(infile)
    writer = gguf.GGUFWriter(outfile, arch)
    ggufio.copy_kv(reader, writer)
    st = ExportStats()
    for t in reader.tensors:
        data = np.asarray(t.data)
        base = t.name[: -len(".weight")] if t.name.endswith(".weight") else None
        if (ggufio.target_kind(t.name) == "dense" and base in factors
                and t.tensor_type == gguf.GGMLQuantizationType.F32):
            U, s, Vt = factors[base]
            st.factorized += 1
            st.orig_params += data.size
            st.new_params += U.size + s.size + Vt.size
            writer.add_tensor(base + ".svd_u", U, raw_dtype=gguf.GGMLQuantizationType.F32)
            writer.add_tensor(base + ".svd_s", s, raw_dtype=gguf.GGMLQuantizationType.F32)
            writer.add_tensor(base + ".svd_vt", Vt, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            if ggufio.target_kind(t.name):
                st.orig_params += data.size
                st.new_params += data.size
            writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return st


def _fmt(st: ExportStats, arch: str) -> str:
    ratio = st.new_params / st.orig_params if st.orig_params else 1.0
    return (f"[export] arch={arch} factorized={st.factorized} kept_dense={st.kept_dense} "
            f"params {st.orig_params/1e6:.1f}M -> {st.new_params/1e6:.1f}M ({ratio:.2f}x) "
            f"max_err={st.max_err:g}")


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="factorize a GGUF's linears via SVD")
    ap.add_argument("infile")
    ap.add_argument("outfile")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rank", type=int, help="fixed rank")
    g.add_argument("--frac", type=float, help="rank as fraction of min(out,in)")
    g.add_argument("--energy", type=float, help="keep singular-value energy fraction (e.g. 0.99)")
    args = ap.parse_args(argv)

    if args.rank is not None:
        policy = RankPolicy("fixed", args.rank)
    elif args.frac is not None:
        policy = RankPolicy("frac", args.frac)
    elif args.energy is not None:
        policy = RankPolicy("energy", args.energy)
    else:
        policy = RankPolicy("full")

    _, arch = ggufio.open_reader(args.infile)
    st = export(args.infile, args.outfile, policy)
    print(_fmt(st, arch), file=sys.stderr)


if __name__ == "__main__":
    main()
