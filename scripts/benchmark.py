"""Full benchmark matrix: size / perplexity / accuracy / speed for the baseline
and every compression variant, so the tradeoffs are visible side by side.

    .venv/bin/python scripts/benchmark.py [family]      # family: smol (default) | qwen

Writes runs/benchmark-<family>.csv and prints a markdown table. Each metric is
isolated so one failure (e.g. a missing dataset) doesn't sink the whole run.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distillrank import evaltools  # noqa: E402

D = "data/eval"
PPL_FILE = f"{D}/wiki.micro.raw"
HS_FILE = f"{D}/hellaswag_val_full.txt"
WG_FILE = f"{D}/winogrande-debiased-eval.csv"
CTX = 256
HS_TASKS = 400
WG_TASKS = 400

# (label, gguf, calibration note). base first, then variants by data used.
MANIFEST = {
    "smol": [
        ("base (f32, uncompressed)", "out/smollm2-135m-base-f32.gguf", "—"),
        ("plain SVD", "runs/smollm2-plain-budget0.6/model.gguf", "0 tokens"),
        ("analytic MC", "runs/smollm2-analytic-budget0.6/model.gguf", "0 tokens"),
        ("random-token prior", "runs/smollm2-randtok-budget0.6/model.gguf", "0 tokens"),
        ("data 2-seq", "runs/smollm2-data2-budget0.6/model.gguf", "512 tokens"),
        ("data 24-seq (Phase-2)", "runs/smollm2-budget0.6/model.gguf", "12k tokens"),
        ("hybrid (analytic+512tok)", "runs/smollm2-hybrid-budget0.6/model.gguf", "512 tokens"),
        ("hybrid + KD finetune", "runs/smollm2-hybrid-ft-budget0.6/model.gguf", "512 tok + KD"),
        # two-sided (IO-SVD) arms — matched-calibration pairs isolate the influence effect
        ("input-only (data 8-seq)", "runs/smollm2-inputonly8/model.gguf", "4k tokens"),
        ("data IO-SVD (two-sided)", "runs/smollm2-influence-budget0.6/model.gguf", "4k tokens"),
        ("zero-data IO-SVD (novel)", "runs/smollm2-zerofisher-budget0.6/model.gguf", "0 tokens"),
    ],
    "qwen": [
        ("base (f32, uncompressed)", "out/qwen2.5-0.5b-base-f32.gguf", "—"),
        ("random-token prior", "runs/qwen05-randtok-budget0.6/model.gguf", "0 tokens"),
        ("data 2-seq", "runs/qwen05-data2-budget0.6/model.gguf", "512 tokens"),
        ("hybrid (analytic+512tok)", "runs/qwen05-hybrid-budget0.6/model.gguf", "512 tokens"),
    ],
}


def _ratio(gguf: str) -> float | None:
    """param_ratio from the sibling results.json if the run wrote one."""
    rj = Path(gguf).parent / "results.json"
    if rj.exists():
        try:
            return json.loads(rj.read_text()).get("param_ratio")
        except Exception:
            return None
    return None


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        print(f"    ! {fn.__name__} failed: {e}", file=sys.stderr)
        return None


def bench(label: str, gguf: str, note: str) -> dict:
    if not Path(gguf).exists():
        print(f"[skip] {label}: {gguf} missing", file=sys.stderr)
        return {"variant": label, "calib": note, "missing": True}
    print(f"[bench] {label}  ({gguf})", file=sys.stderr)
    row = {"variant": label, "calib": note,
           "size_mb": round(evaltools.size_bytes(gguf) / 1e6, 1),
           "param_ratio": _ratio(gguf)}
    row["ppl"] = _try(evaltools.perplexity, gguf, PPL_FILE, CTX)
    if Path(HS_FILE).exists():
        row["hellaswag_%"] = _try(evaltools.hellaswag, gguf, HS_FILE, HS_TASKS)
    if Path(WG_FILE).exists():
        row["winogrande_%"] = _try(evaltools.winogrande, gguf, WG_FILE, WG_TASKS)
    row.update(_try(evaltools.speed, gguf) or {})
    print(f"    {json.dumps({k: v for k, v in row.items() if k not in ('variant', 'calib')})}",
          file=sys.stderr)
    return row


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.1f}" if abs(v) >= 100 else f"{v:.3g}"
    return str(v)


def main():
    fam = sys.argv[1] if len(sys.argv) > 1 else "smol"
    rows = [bench(*m) for m in MANIFEST[fam]]
    rows = [r for r in rows if not r.get("missing")]

    cols = ["variant", "calib", "size_mb", "param_ratio", "ppl",
            "hellaswag_%", "winogrande_%", "pp_tps", "tg_tps"]
    cols = [c for c in cols if any(c in r for r in rows)]
    out = Path(f"runs/benchmark-{fam}.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    hdr = {"variant": "variant", "calib": "calib data", "size_mb": "size MB",
           "param_ratio": "params", "ppl": "PPL↓", "hellaswag_%": "HellaSwag↑",
           "winogrande_%": "Winogrande↑", "pp_tps": "prefill tok/s↑",
           "tg_tps": "decode tok/s↑"}
    print("\n| " + " | ".join(hdr[c] for c in cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
