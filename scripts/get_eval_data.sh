#!/usr/bin/env bash
# Fetch evaluation datasets in the formats llama-perplexity expects.
#   data/eval/wiki.test.raw            (perplexity)
#   data/eval/hellaswag_val_full.txt   (--hellaswag)
#   data/eval/winogrande-debiased-eval.csv  (--winogrande)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="$ROOT/data/eval"
mkdir -p "$D"

if [ ! -f "$D/wiki.test.raw" ]; then
    tmp="$(mktemp -d)"
    curl -sL -o "$tmp/w.zip" \
        https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    unzip -q "$tmp/w.zip" -d "$tmp"
    cp "$tmp"/wikitext-2-raw/wiki.test.raw "$D/wiki.test.raw"
    rm -rf "$tmp"
fi

[ -f "$D/hellaswag_val_full.txt" ] || curl -sL -o "$D/hellaswag_val_full.txt" \
    https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt

[ -f "$D/winogrande-debiased-eval.csv" ] || curl -sL -o "$D/winogrande-debiased-eval.csv" \
    https://huggingface.co/datasets/ggml-org/ci/resolve/main/winogrande-debiased-eval.csv

echo "Eval data in $D:"; ls -la "$D"
