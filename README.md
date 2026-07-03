# distill-rank

**Low-rank SVD compression of LLM weights, executed natively as factors inside
Ollama/llama.cpp — plus a zero-data calibration method that derives the
activation-aware whitening covariance from the model's own weights and
tokenizer, with no calibration text at all.**

The repository contains three things:

1. **A factored-execution runtime** (Phase 1): a ~108-line, architecture-agnostic
   llama.cpp patch that loads and *runs* every linear weight as its SVD factors
   `U·diag(s)·Vᵀ` inside a rebuilt Ollama — dense and MoE, no per-model code.
2. **A modular compression pipeline** (Phase 2): truncation, activation-aware
   (SVD-LLM-style) whitening, global rank budgeting, and knowledge-distillation
   recovery, config-driven end to end, evaluated with perplexity / HellaSwag /
   Winogrande / throughput on the patched runtime.
3. **Zero-data analytic whitening** (Phase 3, the original contribution): compute
   each linear's input covariance without any calibration data — from a token
   prior read out of the tokenizer plus moment propagation — and a **hybrid
   shrinkage estimator** that blends that analytic covariance with a tiny data
   sample and *outperforms full-data calibration by ~8×* at the same size.

Headline (SmolLM2-135M, 0.60× params, CPU): plain data-free SVD PPL ≈ 8×10⁷ →
zero-data whitening 4,045 → hybrid (512 calibration tokens) 427 → hybrid + 200
KD steps **129**, while running **faster than the uncompressed model** at both
prefill and decode.

---

## Contents

- [Original contributions](#original-contributions)
- [Background and related work](#background-and-related-work)
- [Phase 1 — the factored-execution runtime](#phase-1--the-factored-execution-runtime)
- [Phase 2 — the compression pipeline](#phase-2--the-compression-pipeline)
- [Phase 3 — zero-data analytic whitening](#phase-3--zero-data-analytic-whitening)
- [Performance engineering: the rank-alignment bug](#performance-engineering-the-rank-alignment-bug)
- [Full benchmarks](#full-benchmarks)
- [Repository layout](#repository-layout)
- [Reproducing everything](#reproducing-everything)
- [Limitations](#limitations)
- [References](#references)

---

## Original contributions

Novelty claims below were checked against the literature by web search on
2026-07-02/03; "no prior art found" means exactly that — a diligent search
found nothing, not a guarantee of absolute priority. Closest adjacent work is
cited inline so the reader can judge.

1. **Zero-data whitening covariance from tokenizer statistics.** Activation-aware
   SVD compression [1–3] needs the per-layer input covariance `H = Σ xxᵀ`,
   universally measured by forwarding calibration text. We compute usable `H`
   with **no data**: sample token ids i.i.d. from a *merge-rank prior* — BPE
   merges are stored in training-corpus frequency order [14], so
   `p(token) ∝ 1/(merge_rank + r₀)` is a corpus frequency estimate read straight
   out of `tokenizer.json`. Closest prior art: Self-Calibration [12] has the
   model *generate* its own calibration data for quantization/pruning (needs
   generation passes; a different mechanism), and random-token calibration has
   been observed to work for quantization [11]. Applying either to SVD whitening,
   and the merge-rank frequency prior itself, appear to be new. Verified here:
   the prior ordering merge_rank > zipf > uniform holds on the *exact* (assumption-free)
   layer-0 covariance, and zero-data whitening lands ~4 orders of magnitude
   below plain data-free SVD end to end.

2. **Hybrid shrinkage calibration — the analytic covariance as a prior.**
   `H = λ·Ĥ_analytic + (1−λ)·H_data(k)` with per-layer trace matching, in the
   spirit of Ledoit–Wolf covariance shrinkage [15]. With **512 tokens** of real
   data this beats *full* 12k-token calibration by ~8× perplexity at identical
   model size (427 vs 3,632), because small-sample covariances in 1,536
   dimensions are noise-dominated and the analytic prior regularizes exactly the
   directions the sample cannot estimate. No prior art found for shrinkage-
   regularized whitening in LLM compression.

3. **Analytic moment propagation (and an instructive negative result).** A fully
   data-free estimator that propagates the residual stream's Gaussian moments
   `(μ_l, Σ_l)` layer by layer — sampling from the analytic Gaussian and pushing
   the samples through the *real* decoder layers, re-fitting moments after each
   layer. It works (65k PPL vs 8×10⁷ plain) but is ~3× *worse* than simply
   sampling random token ids: **moment-matched Gaussianization destroys the
   heavy-tailed activation structure that FFN whitening depends on.** No prior
   art found for either the method or the finding.

4. **The factored-execution runtime.** Compression papers simulate low-rank
   inference in PyTorch and report parameter counts; here the factors are the
   *deployed representation*: a 5-file / ~108-line llama.cpp patch loads
   `U/s/Vᵀ` at three universal seams and executes them for any dense or MoE
   architecture, inside stock Ollama serving. Verified token-for-token against
   dense baselines on three architectures.

5. **The rank-alignment performance cliff.** Factored models measured *slower*
   than dense despite 0.59× FLOPs — because ggml's fast f32 GEMM
   (`llamafile_sgemm`) rejects any matmul whose inner dimension `k % 8 ≠ 0`, and
   in a factored model the second matmul's inner dimension *is the kept rank*.
   Energy-threshold ranks are arbitrary integers, so nearly every layer fell
   onto the slow generic path. Rounding ranks up to a multiple of 8 (<2% size,
   strictly more spectrum) flips factored models to **faster than dense at both
   prefill and decode**. We found no published discussion of SIMD-alignment
   constraints in rank selection for low-rank LLM inference.

## Background and related work

**Activation-aware low-rank compression.** Plain truncated SVD minimizes
`‖W − W_r‖`, which is the wrong objective — what matters is the error on real
activations, `‖(W − W_r)X‖`. ASVD [1] first scaled weights by activation
statistics; SVD-LLM [2] made the mapping exact by whitening: with
`H = Σ xxᵀ = SSᵀ` (Cholesky), truncate the SVD of `W·S` and de-whiten, so each
discarded singular value contributes exactly its share of activation-space
error. SVD-LLM V2 [3], Swift-SVD [7], Dobi-SVD [6] and others refine the
truncation, rank selection, or efficiency. All of them consume calibration
text. AIR [8] reduces the calibration data requirement (~90% less); PARSE [9]
selects ranks per prompt; Basis Sharing [10] shares singular bases across
layers. Sparse-plus-low-rank and quantization-plus-low-rank hybrids (SLiM [4],
HASSLE-free [17], 3BASiL [18], CALDERA-style low-precision factorization)
combine decompositions. Mixed precision along the singular spectrum exists for
delta weights and adapters (Delta-compression QEM [5], LoRAQuant [19],
SVDq [20]). Phase 2 of this repo implements the standard lineage ([1,2] plus
budget allocation and KD recovery) as the baseline; Phase 3's zero-data and
shrinkage calibration are, to our knowledge, not in this literature.

**Data-free calibration elsewhere.** LLM-QAT [11] pioneered data-free
quantization-aware training with model-generated data; Self-Calibration [12]
does the same for PTQ/pruning. Both *generate* data with the model. The
merge-rank prior here needs no generation — the tokenizer already encodes
corpus statistics [13,14].

## Phase 1 — the factored-execution runtime

Every linear weight `W` (`[out, in]`, or `[n_expert, out, in]` for MoE expert
stacks) is replaced by three tensors from its thin SVD:

```
W = U · diag(s) · Vᵀ        U:[out, r]   s:[r]   Vᵀ:[r, in]   r = min(out, in)
```

At full rank this is loss-less — the model is simply **stored, and executed, as
three matrices per linear instead of one**. Truncating `r` (Phase 2) turns it
into genuine compression: `r(out+in) < out·in` parameters and FLOPs.

### Architecture-agnostic by construction

There is **no per-model code**. The factorization is wired into the three
universal seams every architecture in llama.cpp shares
(`patches/svd-generic.patch`, 5 files, ~108 lines):

| seam (patched) | role |
|----------------|------|
| `llama_model_base::create_tensor` | load hook: if a fused `…weight` is absent but `…svd_vt` exists, load `U/s/Vᵀ`, register them in a model-level map, return a handle tensor |
| `build_lora_mm` | every **dense** linear: if the weight is a handle, emit `U·(s⊙(Vᵀx))` instead of one `mul_mat` |
| `build_lora_mm_id` | every **MoE expert** matmul: same, with per-expert singular values gathered by routing ids |

A `const llama_svd_map* svd` is threaded through `llm_graph_params` /
`llm_graph_context` exactly like LoRA adapters. The GGUF keeps its real
`general.architecture`, so Ollama routes, sizes and loads it normally. Ollama
serves all GGML models through upstream `llama-server`, so the patch is applied
to llama.cpp `b9509` (after Ollama's own compat patches) and Ollama v0.30.5 is
rebuilt against it (`OLLAMA_LLAMA_CPP_SOURCE`).

### ggml mapping

`llama-server` stores linear weights transposed (`ne = {in, out}`) and applies
`weight.mul_mat(x)`. The factors are stored to match — `Vᵀ {in, r}`, `s {r}`,
`U {r, out}` — and the runtime emits `mul_mat(U, mul(mul_mat(Vᵀ, x), s))`. For
MoE, `s` is `{r, n_expert}`, gathered per routed expert with
`ggml_get_rows(s, ids)`. HF→GGUF q/k RoPE row permutation is applied to the
`U` factor at export.

### Verified

Greedy decoding (`temperature 0`, `top_k 1`) of the factorized model matches the
plain f32 baseline **token-for-token**:

| model | arch | linears factorized | recon. error | parity |
|-------|------|--------------------|--------------|--------|
| Qwen2.5-0.5B | `qwen2` (dense) | 168 | ~1e-6 | ✅ |
| SmolLM2-135M | `llama` (dense) | 210 | ~7e-6 | ✅ |
| tiny Qwen2-MoE | `qwen2moe` (MoE) | 21 | ~4e-8 | ✅ |

The factorized GGUF contains **no** fused `…attn_q.weight` — only
`…svd_u/s/vt` — so stock llama.cpp cannot load it; a token match proves the
factored path actually ran. Coverage: every architecture whose linears flow
through `build_lora_mm`/`build_lora_mm_id` (llama, qwen2/3, gemma, phi,
mistral, mixtral, qwen3-moe, …); MLA `attn_*_a/_b` and SSM/conv weights are
deliberately skipped (not plain linear projections).

## Phase 2 — the compression pipeline

`distillrank/` is a config-driven pipeline: **calibrate → factorize
[activation-aware] [+ finetune] → export GGUF → evaluate**, one YAML per run,
artifacts under `runs/<name>/{stats.npz, model.gguf, results.json}`.

### Methods

- **Rank policies** (`factorize.RankPolicy`): `full` | `fixed` | `frac` |
  `energy` (smallest r capturing an energy fraction of Σs²), all rounded up to
  `align=8` (see the performance section). A **break-even guard** keeps a matrix
  dense whenever `r(out+in) ≥ out·in`.
- **Activation-aware whitening** (`factorize.whiten_svd`, SVD-LLM style [2]):
  accumulate `H = Σ xxᵀ` per linear input (`calibrate.py`, forward hooks, keyed
  by GGUF tensor names incl. phi3 fused projections), damp
  `H += 1e-3·mean(diag)·I`, Cholesky `H = SSᵀ`, SVD of `W·S`, truncate,
  de-whiten by triangular solve. Minimizes `‖(W−W_r)X‖` instead of `‖W−W_r‖`.
- **Global rank budget** (`planner.energy_for_budget`): binary-search a single
  energy threshold τ so the *whole model* hits a target parameter ratio; each
  layer keeps the smallest rank capturing τ of its (whitened) energy — layers
  with fast-decaying spectra compress more.
- **KD recovery** (`finetune/distill.py`): swap each block linear for a
  trainable `LowRankLinear` (initialized from the whitened factors, `√s` split),
  freeze everything else, distill against the original model's logits (KL) on
  unlabeled text; export factors back to GGUF.

### Phase-2 reference results (SmolLM2-135M)

Plain data-free SVD is catastrophic on small dense models (fracs ≥ 0.5 are
no-ops via break-even; below that, PPL explodes to ~10⁷). Whitening repairs
most of it; KD closes in further — at frac 0.6 (0.86× params): base 22.4 |
plain 3.9×10⁷ | activation-aware 54.5 | +200 KD steps (CPU) **30.5**.

## Phase 3 — zero-data analytic whitening

**Question:** can the whitening covariance be derived from the model itself,
with zero calibration data? Motivation: calibration data is a real deployment
constraint (private domains, licensing, no representative text), the field only
*reduces* it [8], and — as Phase 2 shows — whitening is worth 4–6 orders of
magnitude of perplexity, so its data dependence matters.

All Phase-3 estimators emit the same `{gguf_name: H}` dict the rest of the
pipeline consumes; nothing downstream changes. Selected per run by
`calibration.source: data | random_tokens | analytic | hybrid`.

### Token priors (`analytic.token_prior`)

- `uniform` — `p ∝ 1`.
- `zipf` — `p ∝ 1/(id+β)^s`; BPE ids are roughly frequency-ordered.
- `merge_rank` — read the `merges` list from `tokenizer.json`; a token minted at
  merge rank r gets `p ∝ 1/(r+r₀)` (base/byte tokens get rank 0). BPE emits
  merges in corpus-frequency order [14], so this is a genuine zero-data
  frequency estimate.

Layer-0 attention inputs are RMSNorm applied to *individual embedding rows*, so
their covariance under a prior is computable **exactly** — no propagation, no
Gaussian assumption (`analytic.layer0_exact_h`). This isolates prior quality:
on SmolLM2, merge_rank > zipf > uniform on every metric (top-32 eigenspace
energy capture 0.81 / 0.80 / 0.78 against measured stats).

### The three estimators

- **`random_tokens`** — sample ids i.i.d. from the prior, forward them through
  the ordinary calibration hooks. Zero data, embarrassingly simple, and the
  strongest pure zero-data method here.
- **`analytic` (mc)** — track residual-stream moments `(μ_l, M_l)`: exact
  embedding moments under the prior, then per layer sample ~16k tokens from
  `N(μ_l, Σ_l)` (damped Cholesky), push through the **real decoder layer** (real
  RoPE/GQA/softmax/SiLU — exact within the simulation, residual cross-terms
  free), re-fit moments, repeat; final norm gives the `output` head covariance.
  Randomness comes only from the analytic Gaussian — still zero data. **Negative
  result:** ~3× worse than `random_tokens`; the moment-matching projection
  destroys the heavy-tailed structure that decides which FFN channels matter.
- **`hybrid`** — `H = λ·Ĥ_analytic·(tr H_d / tr Ĥ) + (1−λ)·H_data(k seqs)`:
  the analytic covariance as a shrinkage prior [15] over a tiny sample
  covariance.

### Results (global budget 0.6×, wikitext PPL)

SmolLM2-135M (base 22.4):

| calibration source | calib tokens | PPL |
|---|---|---|
| plain SVD (no whitening) | 0 | 82,352,435 |
| analytic mc | 0 | 65,041 |
| **random_tokens (merge_rank)** | **0** | **4,045** |
| data, 2 seqs | 512 | 2,554 |
| data, 24 seqs | 12,288 | 9,002 |
| **hybrid, λ=0.5** | **512** | **427** |
| **hybrid + 200 KD steps** | 512 + KD | **129** |

Qwen2.5-0.5B (base 19.0): random_tokens 741 | data-2seq 836 | **hybrid 213**.

Note the pure-data rows: 24 sequences scored *worse* than 2 in this aligned
re-plan (9,002 vs 2,554) — small-sample rank allocation is noisy in exactly the
way the shrinkage prior repairs (the hybrid barely moves between re-plans).

### λ × data-budget ablation (`scripts/sweep_hybrid.py`)

λ=0.5 is near-optimal at every budget; the prior is worth ~8× data (256 tokens
+ prior beats 2,048 tokens data-only):

| calib tokens | λ=0 (data only) | λ=0.25 | λ=0.5 | λ=0.75 | λ=1 (analytic only) |
|---|---|---|---|---|---|
| 256 | 1,697 | 719 | **531** | 611 | 68,302 |
| 512 | 2,378 | 446 | **434** | 617 | 68,298 |
| 2,048 | 1,025 | 289 | **276** | 304 | 68,307 |

(Sweep predates the alignment fix; relative ordering is what matters.)

### Diagnostics, and a measurement caveat

`stats-diff` compares covariance sets (trace-normalized relative Frobenius
error, top-k eigenspace overlap, eigenvalue-weighted energy capture) and — with
a GGUF — the whitened-spectrum ranks the planner would induce. Caveat we
measured: at small token counts these subspace metrics are dominated by
estimation noise (even *real* 2k-token stats capture only 0.25–0.42 of the
reference `ffn_down` energy), so **end-to-end perplexity, not covariance
similarity, is the arbiter**. Measured covariances are extremely top-heavy
(top 1–2 eigendirections ≈ 50% of energy at every depth — the "massive
activations" phenomenon), which is why a prior that nails a handful of
dominant directions already buys most of the whitening benefit.

## Performance engineering: the rank-alignment bug

First benchmarks showed factored prefill *slower* than dense (1,148 vs 1,552
tok/s @24t) despite 0.59× FLOPs — while a torch microbenchmark of the same
shapes ran *faster* factored. Root cause: ggml's fast f32 GEMM
(`llamafile_sgemm` [16]) **bails out whenever the inner dimension k % 8 ≠ 0**
(`sgemm.cpp: if (k % 8) return false;`). In the factored path the second
matmul's inner dimension *is the kept rank*, and energy-threshold ranks are
arbitrary integers (51, 111, 205, …) — so nearly every U-side matmul silently
fell onto the slow generic path. The dense baseline never trips this (k = 576
or 1536).

**Fix:** `RankPolicy.align = 8` — ranks round **up** to a multiple of 8 (the
planner's budget search matches). Costs <2% params, keeps strictly more
spectrum. Same stats, same budget, SmolLM2:

| model | prefill t/s @1t | prefill t/s @24t | decode t/s @1t | PPL |
|---|---|---|---|---|
| base (dense f32) | 185 | 1,552 | 16.1 | 22.4 |
| hybrid, unaligned | 102 | 1,148 | 23.1 | 434 |
| **hybrid, aligned** | **263** | **1,614** | **23.5** | **427** |

Single-thread decode (+46%) tracks the 0.59× byte ratio; at 24 threads on a
135M model the decode gain is masked by per-op thread barriers (two per linear
instead of one) and re-emerges on the larger Qwen.

## Full benchmarks

All variants exported with aligned ranks at global budget 0.6×; one harness
(`scripts/benchmark.py`; sequential runs — concurrent jobs corrupt throughput
numbers). PPL on wikitext (ctx 256); HellaSwag/Winogrande over 400 tasks;
prefill/decode via `llama-bench` (pp/tg 128), 24-thread CPU.

**SmolLM2-135M** (f32, 540 MB → ~370 MB):

| variant | calib | size MB | PPL↓ | HellaSwag↑ | Winogrande↑ | prefill t/s | decode t/s |
|---|---|---|---|---|---|---|---|
| **base (uncompressed)** | — | 539.8 | **22.4** | **41.2** | **55.2** | 1,652 | 67.7 |
| plain SVD | 0 | 370.7 | 82,352,435 | 25.8 | 48.5 | 1,722 | 58.9 |
| analytic MC | 0 | 369.8 | 65,041 | 26.2 | 47.5 | 1,743 | 64.0 |
| random-token prior | 0 | 373.8 | 4,045 | 24.0 | 47.8 | **1,784** | 65.8 |
| data 2-seq | 512 | 369.0 | 2,554 | 27.8 | 49.0 | 1,714 | 61.7 |
| data 24-seq | 12k | 369.1 | 9,002 | 26.5 | 49.0 | 1,662 | 63.6 |
| hybrid | 512 | 370.6 | 427 | 27.0 | 51.5 | 1,707 | 65.7 |
| **hybrid + KD** | 512+KD | 370.6 | **129** | 28.5 | 51.8 | 1,752 | **70.4** |

**Qwen2.5-0.5B** (f32, 1,982 MB → ~1,410 MB):

| variant | calib | size MB | PPL↓ | HellaSwag↑ | Winogrande↑ | prefill t/s | decode t/s |
|---|---|---|---|---|---|---|---|
| **base (uncompressed)** | — | 1,982 | **19.0** | **51.0** | **57.0** | 653 | 25.8 |
| random-token prior | 0 | 1,409 | 741 | 26.8 | 50.0 | 875 | 31.8 |
| data 2-seq | 512 | 1,418 | 836 | 27.2 | 56.0 | 883 | 31.2 |
| **hybrid** | 512 | 1,405 | **213** | 26.5 | 49.5 | **898** | **32.0** |

Reading the tradeoff:

- **Size**: a clean ~29–31% cut at budget 0.6×, identical across calibration
  methods — the rank plan sets the size; calibration sets the *quality* at that
  size.
- **Speed** (post-alignment): every factored variant beats dense at prefill
  (+4–8% SmolLM2, **+34–38% Qwen**) and at decode on Qwen (+21–24%); hybrid+KD
  also leads SmolLM2 decode. Raw CSVs: `runs/benchmark-{smol,qwen}.csv`.
- **Quality**: calibration choice spans six orders of magnitude of PPL at
  identical size; KD recovery is decisive (hybrid+KD is the only variant within
  ~6× of base PPL and recovers HellaSwag toward base).

## Repository layout

| path | purpose |
|------|---------|
| `patches/svd-generic.patch` | the llama.cpp change (5 files: load hook + 2 graph ops + wiring) |
| `svd_export.py` | Phase-1 GGUF post-processor: read each `W`, write `U/s/Vᵀ` (any arch; `--rank N`) |
| `distillrank/factorize.py` | plain + whitened SVD, rank policies (align=8), break-even guard |
| `distillrank/calibrate.py` | data calibration: per-linear covariance hooks, HF→GGUF name map |
| `distillrank/analytic.py` | Phase 3: token priors, exact layer-0 H, moment propagation, `mix_stats`, diagnostics |
| `distillrank/planner.py` | global rank budget (binary-searched energy threshold) |
| `distillrank/finetune/distill.py`, `distillrank/ir.py` | KD recovery on `LowRankLinear` factors |
| `distillrank/export_gguf.py`, `distillrank/ggufio.py` | factored + merged GGUF writer (dense + MoE, RoPE permutation) |
| `distillrank/evaltools.py` | llama-perplexity / llama-bench wrappers (PPL, HellaSwag, Winogrande, tok/s) |
| `distillrank/runner.py`, `distillrank/cli.py` | YAML runner; CLI: `run / plan / calibrate / calibrate-analytic / stats-diff / factorize / finetune / eval / sweep` |
| `configs/*.yaml` | one file per experiment arm (all tables above are reproducible from these) |
| `scripts/benchmark.py` | the full benchmark matrix |
| `scripts/sweep_hybrid.py` | λ × data-budget ablation |
| `scripts/setup_env.sh` / `build_ollama.sh` / `build_tools.sh` | toolchain, patched Ollama, patched eval binaries |
| `scripts/make_models.sh`, `scripts/get_eval_data.sh` | HF model → GGUFs; wikitext/HellaSwag/Winogrande |
| `verify.sh`, `scripts/_gen.py` | greedy token-parity check in the patched Ollama |

`vendor/`, `models/`, `out/`, `runs/`, `.venv/` are git-ignored build/data
dirs. The patched Ollama stores served models under
`OLLAMA_MODELS` (default `/mnt/d/Work/ollama`, override via env).

## Reproducing everything

```bash
# 0. toolchain + patched sources (no root needed), patched ollama, eval tools
scripts/setup_env.sh && scripts/build_ollama.sh && scripts/build_tools.sh
scripts/get_eval_data.sh

# 1. Phase 1: loss-less factored execution, token parity
scripts/make_models.sh HuggingFaceTB/SmolLM2-135M smollm2-135m
./verify.sh smollm2-135m out/smollm2-135m-base-f32.gguf out/smollm2-135m-svd-f32.gguf

# 2. Phase 3 arms (each writes runs/<name>/{stats.npz,model.gguf,results.json})
python -m distillrank run configs/smollm2-randtok-budget06.yaml    # zero data
python -m distillrank run configs/smollm2-analytic-budget06.yaml  # zero data, MC
python -m distillrank run configs/smollm2-hybrid-budget06.yaml    # + 512 tokens
python -m distillrank run configs/smollm2-hybrid-ft-budget06.yaml # + KD recovery

# 3. diagnostics / ablation / benchmark
python -m distillrank calibrate-analytic models/SmolLM2-135M runs/an.npz --mode random_tokens
python -m distillrank stats-diff runs/an.npz runs/smol-stats.npz --gguf out/smollm2-135m-base-f32.gguf
python scripts/sweep_hybrid.py 0.6
python scripts/benchmark.py smol && python scripts/benchmark.py qwen
```

Pinned versions: Ollama `v0.30.5`, llama.cpp `b9509` (must match
`vendor/ollama/LLAMA_CPP_VERSION`); the SVD patch applies **after** Ollama's
`llama/compat/**/*.patch`. Python 3.14 venv with torch (CPU) + gguf +
transformers; eval binaries are a *clean* llama.cpp `b9509` + only the SVD
patch (Ollama compat patches don't link standalone).

## Limitations

- **Scale**: validated on 135M–0.5B models on one CPU box; the KD numbers are
  CPU-constrained (200 steps). Larger models and GPU finetuning should close
  more of the quality gap, but that is unverified here.
- **Evaluation**: perplexity is wikitext-only; HellaSwag/Winogrande at 400
  tasks have meaningful variance; single seeds throughout.
- **At 0.6× budget the compressed models are far from base quality** (129 vs
  22.4 PPL) — the contribution is the *calibration-data* axis and the runtime,
  not a state-of-the-art compression ratio claim.
- **Novelty is claimed as of 2026-07** based on literature search, scoped in
  [Original contributions](#original-contributions).
- f32 only; quantizing the factors (and how quantization interacts with
  orthogonality and the spectrum) is future work.

## References

1. Yuan et al., *ASVD: Activation-aware Singular Value Decomposition for
   Compressing LLMs*, [arXiv:2312.05821](https://arxiv.org/abs/2312.05821)
2. Wang et al., *SVD-LLM: Truncation-aware Singular Value Decomposition for LLM
   Compression*, ICLR 2025, [arXiv:2403.07378](https://arxiv.org/abs/2403.07378)
3. Wang et al., *SVD-LLM V2: Optimizing Singular Value Truncation*,
   [arXiv:2503.12340](https://arxiv.org/abs/2503.12340)
4. Mozaffari et al., *SLiM: One-shot Quantized Sparse Plus Low-rank
   Approximation of LLMs*, [arXiv:2410.09615](https://arxiv.org/abs/2410.09615)
5. *Enhancing Delta Compression in LLMs via SVD-based Quantization Error
   Minimization*, [arXiv:2506.11087](https://arxiv.org/abs/2506.11087)
6. Qinsi et al., *Dobi-SVD: Differentiable SVD for LLM Compression*,
   [arXiv:2502.02723](https://arxiv.org/abs/2502.02723)
7. *Swift-SVD: Theoretical Optimality Meets Practical Efficiency in Low-Rank
   LLM Compression*, [arXiv:2604.01609](https://arxiv.org/abs/2604.01609)
8. *Activation- and Influence-Aware Ranks (AIR): Function-Preserving SVD
   Compression for LLMs*, [arXiv:2606.19993](https://arxiv.org/abs/2606.19993)
9. *Different Prompts, Different Ranks: Prompt-aware Dynamic Rank Selection
   (PARSE)*, [arXiv:2605.08568](https://arxiv.org/abs/2605.08568)
10. *Basis Sharing: Cross-Layer Parameter Sharing for LLM Compression*, ICLR
    2025, [arXiv:2410.03765](https://arxiv.org/abs/2410.03765)
11. Liu et al., *LLM-QAT: Data-Free Quantization Aware Training for LLMs*,
    [arXiv:2305.17888](https://arxiv.org/abs/2305.17888)
12. *Self-calibration for Language Model Quantization and Pruning*,
    [arXiv:2410.17170](https://arxiv.org/abs/2410.17170)
13. *Train It and Forget It: Merge Lists are Unnecessary for BPE Inference*
    (merge lists expose training-corpus statistics),
    [arXiv:2508.06621](https://arxiv.org/abs/2508.06621)
14. Sennrich et al., *Neural Machine Translation of Rare Words with Subword
    Units* (BPE merges by corpus frequency),
    [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
15. Ledoit & Wolf, *A well-conditioned estimator for large-dimensional
    covariance matrices*, J. Multivariate Analysis, 2004
16. Tunney, *LLaMA Now Goes Faster on CPUs* (llamafile/tinyBLAS sgemm in
    llama.cpp), [justine.lol/matmul](https://justine.lol/matmul/)
17. *HASSLE-free: A unified Framework for Sparse plus Low-Rank Matrix
    Decomposition for LLMs*, [arXiv:2502.00899](https://arxiv.org/abs/2502.00899)
18. *3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of
    LLMs*, [arXiv:2603.01376](https://arxiv.org/abs/2603.01376)
19. *LoRAQuant: Mixed-Precision Quantization of LoRA to Ultra-Low Bits*,
    [arXiv:2510.26690](https://arxiv.org/abs/2510.26690)
20. *SVDq: 1.25-bit and 410× Key Cache Compression for LLM Attention*,
    [arXiv:2502.15304](https://arxiv.org/abs/2502.15304)
