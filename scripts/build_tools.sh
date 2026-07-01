#!/usr/bin/env bash
# Build the patched llama.cpp eval tools (llama-perplexity, llama-bench).
#
# These need a CLEAN llama.cpp + ONLY patches/svd-generic.patch — NOT the Ollama
# compat patches (those inject Ollama-only symbols that don't link standalone).
# So we keep a second checkout, vendor/llama.cpp-tools, just for the tools.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. .venv/bin/activate                          # provides cmake
LLAMA_TAG="b9509"

if [ ! -d vendor/llama.cpp-tools ]; then
    git clone --depth 1 --branch "$LLAMA_TAG" \
        https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp-tools
fi
cd vendor/llama.cpp-tools
git apply --reverse --check "$ROOT/patches/svd-generic.patch" 2>/dev/null \
    || git apply --whitespace=nowarn "$ROOT/patches/svd-generic.patch"

cmake -B build -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build build --target llama-perplexity llama-bench -j "$(nproc)"

echo "Built: $ROOT/vendor/llama.cpp-tools/build/bin/{llama-perplexity,llama-bench}"
