#!/bin/bash
# Original SmolLM2-135M-Instruct weights (bf16 safetensors) + tokenizer from Hugging Face.
set -e; mkdir -p llm; cd llm
B=https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main
curl -sSL -o model.safetensors $B/model.safetensors; curl -sSL -o tokenizer.json $B/tokenizer.json; curl -sSL -o config.json $B/config.json; ls -l
