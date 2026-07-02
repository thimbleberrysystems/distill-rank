"""Hybrid ablation: how much calibration data buys what, with/without the
analytic prior. Sweeps λ (analytic weight) × k (calibration seqs) at a fixed
global budget and writes runs/hybrid-sweep.csv.

    .venv/bin/python scripts/sweep_hybrid.py [budget=0.6]
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distillrank.runner import run  # noqa: E402

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEQS = [1, 2, 8]

rows = []
for k in SEQS:
    for lam in LAMBDAS:
        name = f"hyb-l{lam}-k{k}"
        cfg = {
            "name": name,
            "model": {"dir": "models/SmolLM2-135M",
                      "base_gguf": "out/smollm2-135m-base-f32.gguf"},
            "method": "activation_aware",
            "rank": {"type": "budget", "value": BUDGET},
            "calibration": {
                "source": "hybrid", "lambda": lam, "mode": "mc",
                "prior": "merge_rank", "samples": 16384, "seqlen": 256,
                "text": "data/eval/wiki.test.raw", "seqs": k, "device": "cpu",
            },
            "eval": {"ppl": "data/eval/wiki.micro.raw", "ctx": 256},
        }
        res = run(cfg)
        rows.append({"lambda": lam, "seqs": k, "calib_tokens": k * 256,
                     "ppl": res.get("ppl"), "param_ratio": res.get("param_ratio")})
        print("ROW", rows[-1], flush=True)

out = Path("runs/hybrid-sweep.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out} ({len(rows)} rows)")
