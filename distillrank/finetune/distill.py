"""Recover quality after truncation by finetuning the low-rank factors.

Build a low-rank student (activation-aware or plain SVD init), freeze everything
except the LowRankLinear factors, and train them to match the original model
(knowledge distillation on unlabeled text) or the LM objective. Device-agnostic.
Returns the finetuned factors keyed by GGUF tensor name, ready for export.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..factorize import RankPolicy
from ..ir import LowRankLinear, extract_factors, make_lowrank


def _device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def distill(model_dir: str, texts: list[str], tokenizer, policy: RankPolicy, *,
            stats: dict | None = None, steps: int = 200, lr: float = 1e-4,
            seqlen: int = 512, kd: bool = True, device: str = "auto") -> dict:
    from transformers import AutoModelForCausalLM

    dev = _device(device)
    student = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    replaced = make_lowrank(student, policy, stats)
    student.to(dev).train()

    for p in student.parameters():
        p.requires_grad_(False)
    trainable = []
    for m in student.modules():
        if isinstance(m, LowRankLinear):
            for p in m.parameters():
                p.requires_grad_(True)
                trainable.append(p)
    print(f"[distill] {len(replaced)} low-rank layers, {sum(p.numel() for p in trainable)/1e6:.1f}M "
          f"trainable params, device={dev}", flush=True)

    teacher = None
    if kd:
        teacher = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(trainable, lr=lr)
    ids = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n_win = max(1, ids.shape[0] // seqlen)

    step = 0
    while step < steps:
        for i in range(n_win):
            if step >= steps:
                break
            batch = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(dev)
            logits = student(batch).logits
            if kd:
                with torch.no_grad():
                    t_logits = teacher(batch).logits
                V = logits.shape[-1]
                loss = F.kl_div(F.log_softmax(logits.reshape(-1, V), -1),
                                F.softmax(t_logits.reshape(-1, V), -1), reduction="batchmean")
            else:
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                       batch[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                print(f"[distill] step {step}/{steps} loss {loss.item():.4f}", flush=True)

    student.eval()
    return extract_factors(student)
