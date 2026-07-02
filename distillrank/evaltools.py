"""Thin wrappers over the patched llama.cpp eval binaries.

These binaries are built from a clean b9509 + patches/svd-generic.patch (see
scripts/build_tools.sh), so they load factorized GGUFs through the same
create_tensor / build_lora_mm hooks as the runtime. Point DR_LLAMA_TOOLS at the
bin dir if it isn't the default.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

_TOOLS = Path(os.environ.get(
    "DR_LLAMA_TOOLS", "vendor/llama.cpp-tools/build/bin")).resolve()


def _bin(name: str) -> str:
    p = _TOOLS / name
    if not p.exists():
        raise FileNotFoundError(f"{name} not found under {_TOOLS} — run scripts/build_tools.sh")
    return str(p)


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + "\n" + r.stderr


def size_bytes(model: str) -> int:
    return os.path.getsize(model)


def perplexity(model: str, text_file: str, ctx: int = 512, threads: int | None = None) -> float:
    cmd = [_bin("llama-perplexity"), "-m", model, "-f", text_file, "-c", str(ctx)]
    if threads:
        cmd += ["-t", str(threads)]
    out = _run(cmd)
    m = re.search(r"PPL\s*=\s*([0-9.]+)", out) or re.search(r"perplexity:\s*([0-9.]+)", out)
    if not m:
        raise RuntimeError(f"could not parse perplexity from:\n{out[-800:]}")
    return float(m.group(1))


def _last_running_score(out: str) -> float:
    """Both --hellaswag and --winogrande print one row per task as
    `<task>\\t<running_accuracy>[%]\\t...`; the final cumulative accuracy is the
    running value on the last such row."""
    accs = re.findall(r"(?m)^\s*\d+\t([0-9]+\.[0-9]+)", out)
    return float(accs[-1]) if accs else float("nan")


def hellaswag(model: str, data_file: str, tasks: int = 400, ctx: int = 1024) -> float:
    """Return accuracy (%) on HellaSwag."""
    out = _run([_bin("llama-perplexity"), "-m", model, "-bf", data_file,
                "--hellaswag", "--hellaswag-tasks", str(tasks), "-c", str(ctx)])
    return _last_running_score(out)


def winogrande(model: str, data_file: str, tasks: int = 0, ctx: int = 1024) -> float:
    """Return accuracy (%) on Winogrande."""
    cmd = [_bin("llama-perplexity"), "-m", model, "-f", data_file, "--winogrande", "-c", str(ctx)]
    if tasks:
        cmd += ["--winogrande-tasks", str(tasks)]
    return _last_running_score(_run(cmd))


def speed(model: str, n_prompt: int = 128, n_gen: int = 128, threads: int | None = None) -> dict:
    """Return {'pp_tps': ..., 'tg_tps': ...} via llama-bench JSON."""
    cmd = [_bin("llama-bench"), "-m", model, "-p", str(n_prompt), "-n", str(n_gen), "-o", "json"]
    if threads:
        cmd += ["-t", str(threads)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    data = json.loads(out[out.index("["):out.rindex("]") + 1])
    res = {}
    for row in data:
        if row.get("n_prompt", 0) and not row.get("n_gen", 0):
            res["pp_tps"] = row["avg_ts"]
        elif row.get("n_gen", 0) and not row.get("n_prompt", 0):
            res["tg_tps"] = row["avg_ts"]
    return res
