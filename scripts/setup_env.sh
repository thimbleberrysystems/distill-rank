#!/usr/bin/env bash
# Reproduce the distill-rank phase-1 toolchain from scratch.
#   - Go (user-local tarball, no root)
#   - cmake + python deps in a venv (no root)
#   - clone Ollama (v0.30.5) and llama.cpp (b9509)
#   - apply Ollama's compat patches + our SVD patch to llama.cpp
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OLLAMA_TAG="v0.30.5"
LLAMA_TAG="b9509"          # must equal vendor/ollama/LLAMA_CPP_VERSION
GO_VERSION="go1.26.4"

# --- Go (user-local) ---
if ! /home/franklynece/.local/go/bin/go version >/dev/null 2>&1; then
    curl -sL "https://go.dev/dl/${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tgz
    rm -rf /home/franklynece/.local/go && mkdir -p /home/franklynece/.local
    tar -C /home/franklynece/.local -xzf /tmp/go.tgz
fi
export PATH="/home/franklynece/.local/go/bin:$PATH"

# --- python venv + deps (cmake comes from the pip 'cmake' wheel) ---
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy gguf safetensors huggingface_hub cmake \
    torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install --quiet transformers sentencepiece

# --- sources ---
mkdir -p vendor
[ -d vendor/ollama ]    || git clone --depth 1 --branch "$OLLAMA_TAG" --recurse-submodules \
    https://github.com/ollama/ollama vendor/ollama
[ -d vendor/llama.cpp ] || git clone --depth 1 --branch "$LLAMA_TAG" \
    https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp

# --- patch llama.cpp: Ollama compat patches first, then our SVD patch ---
pushd vendor/llama.cpp >/dev/null
COMPAT="$ROOT/vendor/ollama/llama/compat"
for p in $(find "$COMPAT" -name '*.patch' | sort); do
    git apply --reverse --check "$p" 2>/dev/null || git apply --whitespace=nowarn "$p"
done
git apply --reverse --check "$ROOT/patches/svd-generic.patch" 2>/dev/null \
    || git apply --whitespace=nowarn "$ROOT/patches/svd-generic.patch"
popd >/dev/null

echo "Setup complete. Next:"
echo "  scripts/build_ollama.sh     # build the patched Ollama"
echo "  scripts/make_models.sh      # download Qwen2.5-0.5B and produce GGUFs"
echo "  ./verify.sh                 # check factorized == baseline"
