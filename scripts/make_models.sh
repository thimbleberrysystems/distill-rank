#!/usr/bin/env bash
# Download an HF model and produce both GGUFs: a plain f32 baseline and the
# SVD-factorized one. Works for any architecture llama.cpp can convert.
#
# Usage:
#   scripts/make_models.sh                              # default: Qwen2.5-0.5B
#   scripts/make_models.sh <hf_repo> <name> [--rank N]  # any model
#   e.g. scripts/make_models.sh HuggingFaceTB/SmolLM2-135M smollm2-135m
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. .venv/bin/activate
export LLAMA_CPP_DIR="$ROOT/vendor/llama.cpp"

HF_REPO="${1:-Qwen/Qwen2.5-0.5B}"
NAME="${2:-qwen2.5-0.5b}"
RANK_ARGS=()
[ "${3:-}" == "--rank" ] && RANK_ARGS=(--rank "${4:?rank value}")

mkdir -p "models/$NAME" out
[ -n "$(ls -A "models/$NAME" 2>/dev/null)" ] || python - "$HF_REPO" "models/$NAME" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],
                  allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*','merges*','vocab*'],
                  local_dir=sys.argv[2])
PY

# 1) plain f32 baseline GGUF
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "models/$NAME" \
    --outfile "out/$NAME-base-f32.gguf" --outtype f32
# 2) SVD-factorized GGUF (post-processed from the baseline)
python svd_export.py "out/$NAME-base-f32.gguf" "out/$NAME-svd-f32.gguf" "${RANK_ARGS[@]}"

echo "Produced out/$NAME-base-f32.gguf and out/$NAME-svd-f32.gguf"
echo "Verify: ./verify.sh $NAME out/$NAME-base-f32.gguf out/$NAME-svd-f32.gguf"
