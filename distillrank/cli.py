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

import numpy as np

from . import evaltools, ggufio
from .export_gguf import RankPolicy, export, export_from_factors, _fmt


def _load_stats(path):
    return {k: v for k, v in np.load(path).items()} if path else None


def cmd_run(args):
    from .runner import run_file
    run_file(args.config)


def cmd_plan(args):
    from .planner import energy_for_budget
    tau = energy_for_budget(args.base, args.target)
    print(f"energy threshold {tau:.4f} for target param ratio {args.target} "
          f"(use: factorize --energy {tau:.4f})")


def cmd_calibrate(args):
    from transformers import AutoTokenizer
    from .calibrate import collect_covariance
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    text = open(args.text).read()
    stats = collect_covariance(args.model_dir, [text], tok,
                               seqlen=args.seqlen, max_seqs=args.seqs, device=args.device)
    np.savez_compressed(args.out, **stats)
    print(f"[calibrate] {len(stats)} covariances -> {args.out}", file=sys.stderr)


def cmd_finetune(args):
    from transformers import AutoTokenizer
    from .finetune import distill
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    factors = distill(args.model_dir, [open(args.text).read()], tok, _policy(args),
                      stats=_load_stats(args.stats), steps=args.steps, lr=args.lr,
                      seqlen=args.seqlen, kd=not args.sft, device=args.device)
    st = export_from_factors(args.base, args.outfile, factors)
    ratio = st.new_params / max(st.orig_params, 1)
    print(f"[finetune] factorized {st.factorized} layers -> {args.outfile} ({ratio:.2f}x)", file=sys.stderr)


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
    st = export(args.infile, args.outfile, _policy(args), stats=_load_stats(args.stats),
                merge=args.merge)
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
        st = export(args.base, out, policy, stats=_load_stats(args.stats))
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

    r = sub.add_parser("run"); r.add_argument("config"); r.set_defaults(func=cmd_run)

    pl = sub.add_parser("plan")
    pl.add_argument("base"); pl.add_argument("target", type=float, help="target param ratio, e.g. 0.6")
    pl.set_defaults(func=cmd_plan)

    c = sub.add_parser("calibrate")
    c.add_argument("model_dir"); c.add_argument("text"); c.add_argument("out")
    c.add_argument("--seqlen", type=int, default=512); c.add_argument("--seqs", type=int, default=128)
    c.add_argument("--device", default="auto")
    c.set_defaults(func=cmd_calibrate)

    f = sub.add_parser("factorize")
    f.add_argument("infile"); f.add_argument("outfile")
    f.add_argument("--stats", help="npz of activation covariances (activation-aware)")
    f.add_argument("--merge", action="store_true",
                   help="write reconstructed dense weights (stock-Ollama compatible, same size)")
    g = f.add_mutually_exclusive_group()
    g.add_argument("--rank", type=int); g.add_argument("--frac", type=float); g.add_argument("--energy", type=float)
    f.set_defaults(func=cmd_factorize)

    e = sub.add_parser("eval")
    e.add_argument("model"); _add_eval_flags(e)
    e.set_defaults(func=cmd_eval)

    ft = sub.add_parser("finetune")
    ft.add_argument("model_dir"); ft.add_argument("base"); ft.add_argument("outfile")
    ft.add_argument("text", help="finetuning/distillation text file")
    ft.add_argument("--stats", help="npz covariances (activation-aware init)")
    ft.add_argument("--steps", type=int, default=200); ft.add_argument("--lr", type=float, default=1e-4)
    ft.add_argument("--seqlen", type=int, default=512); ft.add_argument("--device", default="auto")
    ft.add_argument("--sft", action="store_true", help="use LM loss instead of KD from the original")
    gg = ft.add_mutually_exclusive_group()
    gg.add_argument("--rank", type=int); gg.add_argument("--frac", type=float); gg.add_argument("--energy", type=float)
    ft.set_defaults(func=cmd_finetune)

    s = sub.add_parser("sweep")
    s.add_argument("base")
    s.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    s.add_argument("--stats", help="npz of activation covariances (activation-aware)")
    s.add_argument("--out", default="runs/sweep.csv")
    _add_eval_flags(s)
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
