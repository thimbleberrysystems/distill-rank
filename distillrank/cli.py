"""distill-rank command line: factorize, eval, sweep.

    python -m distillrank factorize base.gguf out.gguf --energy 0.99
    python -m distillrank eval out.gguf --ppl data/eval/wiki.test.raw --speed
    python -m distillrank sweep base.gguf --fracs 1.0 0.75 0.5 0.25 \
        --ppl data/eval/wiki.test.raw --speed --out runs/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import evaltools, ggufio
from .export_gguf import RankPolicy, export, _fmt


def _policy(args) -> RankPolicy:
    if getattr(args, "rank", None) is not None:
        return RankPolicy("fixed", args.rank)
    if getattr(args, "frac", None) is not None:
        return RankPolicy("frac", args.frac)
    if getattr(args, "energy", None) is not None:
        return RankPolicy("energy", args.energy)
    return RankPolicy("full")


def cmd_factorize(args):
    _, arch = ggufio.open_reader(args.infile)
    st = export(args.infile, args.outfile, _policy(args))
    print(_fmt(st, arch), file=sys.stderr)


def _eval_one(model: str, args) -> dict:
    row = {"model": Path(model).name, "size_mb": round(evaltools.size_bytes(model) / 1e6, 1)}
    if args.ppl:
        row["ppl"] = round(evaltools.perplexity(model, args.ppl, args.ctx), 4)
    if args.hellaswag:
        row["hellaswag"] = evaltools.hellaswag(model, args.hellaswag, args.hs_tasks)
    if args.winogrande:
        row["winogrande"] = evaltools.winogrande(model, args.winogrande)
    if args.speed:
        row.update(evaltools.speed(model))
    return row


def cmd_eval(args):
    row = _eval_one(args.model, args)
    for k, v in row.items():
        print(f"{k:12s} {v}")


def cmd_sweep(args):
    Path("runs").mkdir(exist_ok=True)
    rows = []
    for frac in args.fracs:
        out = f"runs/{Path(args.base).stem}-frac{frac}.gguf"
        policy = RankPolicy("full") if frac >= 1.0 else RankPolicy("frac", frac)
        st = export(args.base, out, policy)
        print(_fmt(st, "sweep"), file=sys.stderr)
        row = {"frac": frac, "param_ratio": round(st.new_params / max(st.orig_params, 1), 3)}
        row.update(_eval_one(out, args))
        rows.append(row)
        print("  ", row)
    if args.out:
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}", file=sys.stderr)


def _add_eval_flags(p):
    p.add_argument("--ppl", help="perplexity text file (e.g. data/eval/wiki.test.raw)")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--hellaswag", help="hellaswag data file")
    p.add_argument("--hs-tasks", type=int, default=400)
    p.add_argument("--winogrande", help="winogrande csv")
    p.add_argument("--speed", action="store_true", help="run llama-bench tok/s")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="distillrank")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("factorize")
    f.add_argument("infile"); f.add_argument("outfile")
    g = f.add_mutually_exclusive_group()
    g.add_argument("--rank", type=int); g.add_argument("--frac", type=float); g.add_argument("--energy", type=float)
    f.set_defaults(func=cmd_factorize)

    e = sub.add_parser("eval")
    e.add_argument("model"); _add_eval_flags(e)
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("sweep")
    s.add_argument("base")
    s.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    s.add_argument("--out", default="runs/sweep.csv")
    _add_eval_flags(s)
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
