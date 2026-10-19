#!/usr/bin/env bash
# Runs once when the codespace is created. Installs the course libraries (CPU PyTorch to keep it small and fast).
set -e
echo "==> Installing course libraries (takes a few minutes the first time)…"
python -m pip install --user --upgrade pip
python -m pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
echo "==> Installing the Chromium browser for the Playwright (Week 12) examples…"
sudo -E env "PATH=$PATH" python -m playwright install-deps chromium >/dev/null
python -m playwright install chromium
[ -f .env ] || cp .env.example .env
echo "==> Ready. Open CODESPACES.md for how to run each week's code."
