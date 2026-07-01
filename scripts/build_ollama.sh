#!/usr/bin/env bash
# Build the patched Ollama against our local (patched) llama.cpp checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/vendor/ollama"

export PATH="/home/franklynece/.local/go/bin:$PATH"
. "$ROOT/.venv/bin/activate"                       # provides cmake
export OLLAMA_LLAMA_CPP_SOURCE="$ROOT/vendor/llama.cpp"

cmake -B build .
cmake --build build --parallel "$(nproc)"
go build -o ollama .

echo "Built: $ROOT/vendor/ollama/ollama"
