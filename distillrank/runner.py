"""Config-driven pipeline: one YAML describes a full compression run.

Stages (each optional / swappable):
  calibrate -> factorize [activation-aware] [+ finetune] -> export GGUF -> eval

    python -m distillrank run configs/smollm2.yaml

Writes runs/<name>/{stats.npz, model.gguf, results.json}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import evaltools, ggufio
from .export_gguf import RankPolicy, export, export_from_factors, _fmt


def _rank_policy(base_gguf: str, spec: dict, stats: dict | None = None) -> RankPolicy:
    kind = spec.get("type", "full")
    if kind == "budget":
        from .planner import energy_for_budget
        tau = energy_for_budget(base_gguf, float(spec["value"]), stats=stats)
        print(f"[run] budget {spec['value']} -> energy threshold {tau:.4f}")
        return RankPolicy("energy", tau)
    return RankPolicy(kind, float(spec.get("value", 1.0)))


def _calibrate_data(model_dir: str, cal: dict) -> dict:
    from transformers import AutoTokenizer
    from .calibrate import collect_covariance
    tok = AutoTokenizer.from_pretrained(model_dir)
    return collect_covariance(model_dir, [open(cal["text"]).read()], tok,
                              seqlen=cal.get("seqlen", 512), max_seqs=cal.get("seqs", 128),
                              device=cal.get("device", "auto"))


def _calibrate(model_dir: str, cal: dict) -> dict:
    """Dispatch on calibration.source: data (default, unchanged) | analytic |
    random_tokens | hybrid (analytic blended with a small data calibration)."""
    source = cal.get("source", "data")
    if source == "data":
        return _calibrate_data(model_dir, cal)
    from .analytic import analytic_covariance, mix_stats
    kw = dict(prior=cal.get("prior", "merge_rank"), zipf_s=cal.get("zipf_s", 1.0),
              samples=cal.get("samples", 16384), seqlen=cal.get("seqlen", 256),
              rho=cal.get("rho", 0.0), seed=cal.get("seed", 0),
              device=cal.get("device", "cpu"))
    if source in ("analytic", "random_tokens"):
        return analytic_covariance(model_dir, mode=cal.get("mode", "mc")
                                   if source == "analytic" else "random_tokens", **kw)
    if source == "hybrid":
        h_a = analytic_covariance(model_dir, mode=cal.get("mode", "mc"), **kw)
        h_d = _calibrate_data(model_dir, cal)
        return mix_stats(h_a, h_d, float(cal.get("lambda", 0.5)))
    raise ValueError(f"unknown calibration source: {source}")


def run(cfg: dict) -> dict:
    name = cfg["name"]
    out_dir = Path(cfg.get("out_dir", "runs")) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = cfg["model"]
    base_gguf = model["base_gguf"]
    if not Path(base_gguf).exists():
        raise FileNotFoundError(f"base_gguf not found: {base_gguf} (run scripts/make_models.sh)")

    method = cfg.get("method", "plain")           # plain | activation_aware
    rank_spec = cfg.get("rank", {"type": "full"})
    ft_spec = cfg.get("finetune")                 # optional
    needs_stats = method == "activation_aware"

    # --- calibration (only if needed) ---
    stats = None
    if needs_stats:
        cal = cfg["calibration"]
        stats = _calibrate(model["dir"], cal)
        np.savez_compressed(out_dir / "stats.npz", **stats)
        print(f"[run] calibrated {len(stats)} layers ({cal.get('source', 'data')})")

    policy = _rank_policy(base_gguf, rank_spec, stats)
    gguf_out = str(out_dir / "model.gguf")

    # --- factorize (+ optional finetune) ---
    if ft_spec:
        from transformers import AutoTokenizer
        from .finetune import distill
        tok = AutoTokenizer.from_pretrained(model["dir"])
        factors = distill(model["dir"], [open(ft_spec["text"]).read()], tok, policy,
                          stats=stats, steps=ft_spec.get("steps", 200), lr=ft_spec.get("lr", 1e-5),
                          seqlen=ft_spec.get("seqlen", 256), kd=not ft_spec.get("sft", False),
                          device=ft_spec.get("device", "auto"))
        st = export_from_factors(base_gguf, gguf_out, factors)
    else:
        st = export(base_gguf, gguf_out, policy, stats=stats)
    print(_fmt(st, method))

    # --- eval ---
    ev = cfg.get("eval", {})
    res = {"name": name, "method": method, "rank": rank_spec,
           "param_ratio": round(st.new_params / max(st.orig_params, 1), 3),
           "gguf": gguf_out, "size_mb": round(evaltools.size_bytes(gguf_out) / 1e6, 1)}
    if ev.get("ppl"):
        res["ppl"] = round(evaltools.perplexity(gguf_out, ev["ppl"], ev.get("ctx", 512)), 3)
        res["base_ppl"] = round(evaltools.perplexity(base_gguf, ev["ppl"], ev.get("ctx", 512)), 3)
    if ev.get("hellaswag"):
        res["hellaswag"] = evaltools.hellaswag(gguf_out, ev["hellaswag"])
    if ev.get("speed"):
        res.update(evaltools.speed(gguf_out))

    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"[run] {json.dumps(res)}")
    return res


def run_file(path: str) -> dict:
    import yaml
    return run(yaml.safe_load(open(path)))
