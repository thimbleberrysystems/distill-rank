#!/usr/bin/env python3
"""Tiny helper: greedy-generate from a running ollama and print the response.
Usage: _gen.py HOST MODEL PROMPT   (stdlib only, no jq/curl needed)."""
import json
import sys
import urllib.request

host, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
payload = json.dumps({
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0, "seed": 42, "top_k": 1, "num_predict": 48},
}).encode()
req = urllib.request.Request(
    f"http://{host}/api/generate", data=payload,
    headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req, timeout=300).read())["response"], end="")
