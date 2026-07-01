# distill-rank

An experiment in low-rank distillation of LLM weights.

## Phase 1 — store & run a model as SVD factors, inside Ollama, for *any* architecture ✅

Every linear weight `W` (shape `[out, in]`, or `[n_expert, out, in]` for MoE
expert stacks) is replaced by three tensors from its thin SVD:

```
W = U · diag(s) · Vᵀ        U:[out, r]   s:[r]   Vᵀ:[r, in]   r = min(out, in)
```

At full rank this is loss-less, so the model is mathematically unchanged — it is
simply **stored, and executed, as three matrices per linear instead of one**.
Truncating `r` turns this into genuine low-rank compression (`--rank`, Phase 2).

### Architecture-agnostic by construction

There is **no per-model code**. The factorization is wired into the three
universal seams that every architecture in llama.cpp already shares:

| seam (patched) | role |
|----------------|------|
| `llama_model_base::create_tensor` | load hook: if a fused `…weight` is absent but its `…svd_vt` factor exists, load `U/s/Vᵀ`, register them in a model-level map, and return a handle tensor |
| `build_lora_mm` | every **dense** linear: if the weight is a handle, emit `U·(s⊙(Vᵀx))` instead of one `mul_mat` |
| `build_lora_mm_id` | every **MoE expert** matmul: same, with per-expert singular values gathered by the routing ids |

A `const llama_svd_map* svd` is threaded through `llm_graph_params` /
`llm_graph_context` (exactly like LoRA adapters). The GGUF keeps its real
`general.architecture`, so Ollama routes, sizes and loads it normally.

### Why an Ollama rebuild

Ollama serves all GGML models through the upstream **`llama-server`**
(llama.cpp), which has no factorized-linear op. We patch llama.cpp at the seams
above and rebuild Ollama against that source (`OLLAMA_LLAMA_CPP_SOURCE`).

### Verified

Greedy decoding (`temperature 0`, `top_k 1`) of the factorized model matches a
plain f32 baseline **token-for-token**, on multiple architectures:

| model | arch | linears factorized | recon. error | parity |
|-------|------|--------------------|--------------|--------|
| Qwen2.5-0.5B | `qwen2` (dense) | 168 | ~1e-6 | ✅ |
| SmolLM2-135M | `llama` (dense) | 210 | ~7e-6 | ✅ |
| tiny Qwen2-MoE | `qwen2moe` (MoE) | 21 | ~4e-8 | ✅ |

(The MoE case uses a tiny random model — its output is gibberish, but base and
factorized produce *identical* tokens, which is what validates the expert path.)

The factorized GGUF contains **no** fused `…attn_q.weight` — only
`…svd_u / …svd_s / …svd_vt`. Stock llama.cpp cannot load it, so a match proves
the factorized path actually ran.

### Coverage

Works for every architecture whose linears flow through `build_lora_mm` /
`build_lora_mm_id` — i.e. the mainstream dense and MoE models (llama, qwen2/3,
gemma, phi, mistral, mixtral, qwen3-moe, …). The producer deliberately skips
weights that bypass those ops (MLA `attn_*_a/_b`, SSM/conv), since they are not
plain linear projections.

## Layout

| path | purpose |
|------|---------|
| `patches/svd-generic.patch` | the llama.cpp change (5 files: load hook + 2 graph ops + wiring) |
| `svd_export.py` | factorize a GGUF: read each `W`, write `U/s/Vᵀ` (any arch; `--rank N`) |
| `verify.sh`, `scripts/_gen.py` | greedy-parity check for a model pair |
| `scripts/setup_env.sh` | toolchain + sources + patches, from scratch (no root) |
| `scripts/build_ollama.sh` | build patched Ollama against `vendor/llama.cpp` |
| `scripts/make_models.sh` | download an HF model → baseline + factorized GGUFs |

`vendor/`, `models/`, `out/`, `.venv/` are git-ignored build/data dirs.

## Reproduce

```bash
scripts/setup_env.sh                                   # Go + venv + patched sources
scripts/build_ollama.sh                                # build patched ollama
scripts/make_models.sh Qwen/Qwen2.5-0.5B qwen2.5-0.5b  # baseline + factorized GGUFs
./verify.sh qwen2.5-0.5b out/qwen2.5-0.5b-base-f32.gguf out/qwen2.5-0.5b-svd-f32.gguf
# any other model, e.g. a llama-arch one:
scripts/make_models.sh HuggingFaceTB/SmolLM2-135M smollm2-135m
./verify.sh smollm2-135m out/smollm2-135m-base-f32.gguf out/smollm2-135m-svd-f32.gguf
```

## Pinned versions

- Ollama `v0.30.5`, llama.cpp `b9509` (must match `vendor/ollama/LLAMA_CPP_VERSION`).
- The SVD patch is applied **after** Ollama's own `llama/compat/**/*.patch`.

## How the factorized math maps to ggml

`llama-server` stores linear weights transposed (`ne = {in, out}`) and applies
`weight.mul_mat(x)`. The factors are stored to match — `Vᵀ {in, r}`, `s {r}`,
`U {r, out}` — and the runtime emits `mul_mat(U, mul(mul_mat(Vᵀ, x), s))`.
For MoE, `s` is `{r, n_expert}` and is gathered per routed expert with
`ggml_get_rows(s, ids)`.
