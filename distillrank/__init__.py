"""distill-rank: modular low-rank compression + finetune pipeline.

Domains:
  - GGUF/deployment: factorize weights, export factored GGUF, run/eval on the
    patched llama.cpp runtime (this is the data-free path, torch-free).
  - PyTorch/science: calibrate, activation-aware factorize, finetune/distill
    (added in later milestones; pulls in torch/transformers).
"""

__all__ = ["factorize", "ggufio", "evaltools"]
