#!/usr/bin/env bash
# distill-rank verification: greedy-decode an SVD-factorized model and its plain
# baseline in the *patched* Ollama and assert the completions match token-for-token.
# Full-rank SVD is loss-less, so the factorized model must behave like the original.
#
# Usage:
#   ./verify.sh                                  # default: Qwen2.5-0.5B pair
#   ./verify.sh NAME BASE.gguf SVD.gguf          # any model pair
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA="$ROOT/vendor/ollama/ollama"
export OLLAMA_HOST="127.0.0.1:11435"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/mnt/d/Work/ollama}"
export PATH="/home/franklynece/.local/go/bin:$PATH"
mkdir -p "$OLLAMA_MODELS"

NAME="${1:-qwen2.5-0.5b}"
BASE_GGUF="${2:-$ROOT/out/qwen2.5-0.5b-base-f32.gguf}"
SVD_GGUF="${3:-$ROOT/out/qwen2.5-0.5b-svd-f32.gguf}"
[ -x "$OLLAMA" ] || { echo "patched ollama not built at $OLLAMA"; exit 1; }

mk() { # tag, gguf  -> a temp Modelfile with raw template + greedy params
    local mf; mf="$(mktemp)"
    printf 'FROM %s\nTEMPLATE """{{ .Prompt }}"""\nPARAMETER temperature 0\nPARAMETER seed 42\nPARAMETER top_k 1\n' "$2" > "$mf"
    "$OLLAMA" create "$1" -f "$mf" >/dev/null 2>&1
    rm -f "$mf"
}

"$OLLAMA" serve >/tmp/dr-ollama.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
echo "waiting for patched ollama ($OLLAMA_HOST) ..."
for _ in $(seq 1 60); do curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1 && break; sleep 1; done

mk "dr-${NAME}-base" "$BASE_GGUF"
mk "dr-${NAME}-svd"  "$SVD_GGUF"

gen() { python3 "$ROOT/scripts/_gen.py" "$OLLAMA_HOST" "$1" "$2"; }

echo "== $NAME =="
fail=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    b="$(gen "dr-${NAME}-base" "$prompt")"
    s="$(gen "dr-${NAME}-svd"  "$prompt")"
    if [ "$b" == "$s" ]; then
        echo "MATCH   | $prompt"
    else
        echo "MISMATCH| $prompt"; echo "  base: $b"; echo "  svd : $s"; fail=1
    fi
done <<'PROMPTS'
The capital of France is
Once upon a time
The first three prime numbers are
def add(a, b):
Water is made of hydrogen and
PROMPTS

[ "$fail" -eq 0 ] && echo "PASS: $NAME factorized matches baseline." || echo "FAIL: $NAME diverged."
exit $fail
