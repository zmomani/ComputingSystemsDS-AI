#!/usr/bin/env bash
# Install (first time), start Ollama in the background and download a small model that runs on the codespace CPU.
#   bash scripts/start-ollama.sh            -> llama3.2:1b (≈1.3 GB)
#   bash scripts/start-ollama.sh qwen2.5:1.5b
set -e
MODEL="${1:-llama3.2:1b}"
if ! command -v ollama >/dev/null; then
  echo "==> Installing Ollama…"; curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -s http://localhost:11434/api/version >/dev/null; then
  echo "==> Starting the Ollama server…"; (ollama serve > /tmp/ollama.log 2>&1 &); sleep 3
fi
echo "==> Downloading $MODEL…"; ollama pull "$MODEL"
echo "==> Ollama is running at http://localhost:11434 with $MODEL. Try:  ollama run $MODEL \"Hello\""
