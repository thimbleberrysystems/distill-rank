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

## Phase 3 — zero-data analytic whitening (novel)

Activation-aware whitening (M2) needs calibration text. Phase 3 asks: **can the
whitening covariance be derived from the model's own weights, with zero
calibration data?** Prior work reduces calibration data (AIR: "90% less"); as
far as we can tell nobody eliminates it. Two zero-data calibration sources, plus
a hybrid, all emitting the same `stats.npz` the rest of the pipeline consumes:

- **Token prior** (no corpus needed): `uniform`, `zipf` over token id, or
  `merge_rank` — BPE merges are emitted in training-corpus frequency order, so a
  token minted at merge rank *r* gets `p ∝ 1/(r+r₀)`. A genuine zero-data
  frequency estimate read straight out of `tokenizer.json`. On SmolLM2's layer-0
  (where the exact H under a prior is computable with *no* propagation),
  merge_rank > zipf > uniform on every metric (top-32 energy capture 0.81).
- **`random_tokens`**: sample token *ids* from the prior, forward them with the
  ordinary calibration hooks. Zero data, embarrassingly simple.
- **`analytic` (mc)**: propagate the residual stream's Gaussian moments layer by
  layer — sample from the analytic `N(μ_l, Σ_l)`, push through the *real*
  decoder layer, re-fit moments, repeat. Fully data-free, but Gaussianization
  turns out to destroy the heavy-tailed activation structure the FFN whitening
  needs (an informative negative result — see table).
- **`hybrid`**: `H = λ·Ĥ_analytic + (1−λ)·H_data(k seqs)` with trace matching —
  the analytic covariance as a **shrinkage prior** over a tiny sample covariance
  (Ledoit–Wolf flavor).

Results (SmolLM2-135M, global budget 0.6× params, wikitext PPL, base 22.4):

| calibration source | calib tokens | perplexity |
|---|---|---|
| plain SVD (no whitening) | 0 | 21,838,070 |
| analytic mc | 0 | 68,304 |
| **random_tokens (merge_rank prior)** | **0** | **22,682** (~960× better than plain) |
| data, 2 seqs | 512 | 2,378 |
| data, 24 seqs (Phase-2 reference) | 12,288 | 3,632 |
| **hybrid: analytic + 2 seqs, λ=0.5** | **512** | **434** |

Replicated on a second architecture — Qwen2.5-0.5B, budget 0.6×, base PPL 19.0:

| calibration source | calib tokens | perplexity |
|---|---|---|
| random_tokens (merge_rank prior) | 0 | 1,185 |
| data, 2 seqs | 512 | 863 |
| **hybrid: analytic + 2 seqs, λ=0.5** | **512** | **210** |

Two headline findings:

1. **Zero data gets you 3 orders of magnitude.** Whitening against random tokens
   sampled from the tokenizer's own merge-rank prior recovers most of the benefit
   of activation-aware compression without a single byte of calibration text.
2. **The analytic prior beats real calibration.** Blending the analytic
   covariance with just 512 real tokens (PPL 434) outperforms full 12k-token
   calibration (PPL 3,632) by 8×, and its own 512-token data leg (PPL 2,378) by
   5.5× — small-sample covariances are noisy in 1536 dims, and the analytic H
   regularizes exactly the directions the sample can't estimate.

Ablation (`scripts/sweep_hybrid.py` → `runs/hybrid-sweep.csv`): λ=0.5 is
near-optimal at every data budget, and the prior is worth ~8× data — SmolLM2
PPL at budget 0.6, λ across columns:

| calib tokens | λ=0 (data only) | λ=0.25 | λ=0.5 | λ=0.75 | λ=1 (analytic only) |
|---|---|---|---|---|---|
| 256 | 1,697 | 719 | **531** | 611 | 68,302 |
| 512 | 2,378 | 446 | **434** | 617 | 68,298 |
| 2,048 | 1,025 | 289 | **276** | 304 | 68,307 |

```bash
python -m distillrank calibrate-analytic models/SmolLM2-135M runs/analytic.npz \
    --mode random_tokens --prior merge_rank            # zero-data covariances
python -m distillrank stats-diff runs/analytic.npz runs/smol-stats.npz \
    --gguf out/smollm2-135m-base-f32.gguf              # diagnose vs measured stats
python -m distillrank run configs/smollm2-hybrid-budget06.yaml   # best result above
```

New: `analytic.py` (priors, exact layer-0 H, moment propagation, `mix_stats`,
diagnostics); `calibration.source: data | analytic | random_tokens | hybrid` in
run configs; CLI `calibrate-analytic` + `stats-diff`. A caveat we measured: at
tiny sample counts the subspace-overlap metrics are dominated by estimation
noise (even *real* 2k-token stats only capture 0.25–0.42 of the reference
`ffn_down` energy), so end-to-end PPL — not covariance similarity — is the
arbiter.

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
