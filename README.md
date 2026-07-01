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

## Phase 2 — modular compression pipeline

`distillrank/` is a modular pipeline that actually *shrinks* models: truncate the
SVD rank, keep the factored form (which runs on the patched runtime), and measure
the accuracy-vs-speed tradeoff. Config-driven end to end:

```bash
scripts/build_tools.sh            # build llama-perplexity + llama-bench (clean b9509 + svd patch)
scripts/get_eval_data.sh          # wikitext / hellaswag / winogrande
python -m distillrank factorize base.gguf out.gguf --energy 0.99   # or --frac / --rank
python -m distillrank eval out.gguf --ppl data/eval/wiki.test.raw --speed
python -m distillrank sweep base.gguf --fracs 1 .75 .5 .25 --ppl data/eval/wiki.test.raw --speed --out runs/s.csv
```

- **Break-even guard:** a matrix is only factorized when `r(m+n) < mn` — otherwise the
  factored form would be *bigger*, so it's kept dense. (Full-rank factoring never shrinks.)
- **Truncation math:** `r < min(m,n)` gives `r(m+n)` params vs `mn` and squeezes the matmul
  through a width-`r` bottleneck → smaller **and** faster; the cost is approximation error.
- **M1 finding:** plain data-free SVD is brutal on a small dense model. On SmolLM2-135M,
  fracs ≥ 0.5 are no-ops (break-even keeps them dense) and below that quality collapses.
- **M2 — activation-aware (SVD-LLM style):** calibrate on a text set to collect each linear's
  input covariance `H = Σ xxᵀ`, then truncate `W` in the H-metric (factor `W·S` where
  `H = SSᵀ`, truncate, de-whiten). This keeps the directions that matter for real
  activations. At **frac 0.4** on SmolLM2-135M (same 0.60× params):

  | variant | perplexity |
  |---|---|
  | base | 22.4 |
  | plain SVD | 31,495,835 |
  | **activation-aware** | **663.8** (~47,000× better) |

  ```bash
  python -m distillrank calibrate models/SmolLM2-135M data/eval/wiki.test.raw runs/stats.npz --seqs 24
  python -m distillrank factorize base.gguf out.gguf --frac 0.4 --stats runs/stats.npz
  ```

- **M3 — finetune/distill recovery:** replace each block linear with a trainable low-rank
  form (`ir.LowRankLinear`, init from the activation-aware factors), freeze the rest, and
  finetune the factors to match the original model (KD) on unlabeled text. Export the
  finetuned factors back to GGUF (q/k are permuted to GGUF's RoPE basis). At **frac 0.6**
  (0.86× params) on SmolLM2-135M, 200 KD steps on CPU:

  | stage | perplexity |
  |---|---|
  | base | 22.4 |
  | plain SVD | 39,000,000 |
  | activation-aware | 54.5 |
  | **activation-aware + finetune** | **30.5** |

  ```bash
  python -m distillrank finetune models/SmolLM2-135M base.gguf out.gguf data/eval/wiki.test.raw \
      --frac 0.6 --stats runs/stats.npz --steps 200 --device auto   # cuda/mps/cpu
  ```
  (More steps + a GPU + gentler ranks close the rest of the gap.)

- **M4 — config-driven orchestrator + global rank budget.** One YAML describes a full run
  (calibrate → factorize [activation-aware] [+ finetune] → export → eval); the runner executes
  it and writes `runs/<name>/{stats.npz, model.gguf, results.json}`. A `budget` rank mode
  binary-searches a global energy threshold to hit a target parameter ratio (each layer keeps
  the smallest rank capturing that energy — layers with fast-decaying spectra compress more).

  ```bash
  python -m distillrank run configs/smollm2-aa-ft.yaml     # end-to-end from one config
  python -m distillrank plan base.gguf 0.6                 # energy threshold for a 0.6x budget
  ```

Modules: `factorize.py` (plain + `whiten_svd` activation-aware + rank policies + break-even),
`calibrate.py` (per-linear covariance, HF→GGUF name map), `ir.py` (`LowRankLinear`),
`finetune/distill.py` (KD, device-agnostic), `planner.py` (budget search), `runner.py`
(config pipeline), `export_gguf.py` (factored + merged writer, dense + MoE), `evaltools.py`
(llama-perplexity / llama-bench), `ggufio.py`, `cli.py`
(`run`/`plan`/`calibrate`/`factorize`/`finetune`/`eval`/`sweep`; `factorize --merge` writes
reconstructed dense weights that run on **stock** Ollama for measurement).

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
