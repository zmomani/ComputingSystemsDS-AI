#!/usr/bin/env bash
# Lesson and notebook side by side (prototype, Week 2).
#
# Starts a Jupyter server in this codespace and serves the lesson pages from it, so the lesson and its notebook
# share one browser window: the lesson on the left, the real notebook (PyTorch, Transformers, …) in the side panel.
# The buttons on each code example then send that example to a cell of the notebook.
#
#   bash scripts/start-lesson-lab.sh            # port 8888
#   bash scripts/start-lesson-lab.sh 8890       # another port
#
# Leave this terminal open while you work. Stop the server with Ctrl+C.
set -e
cd "$(dirname "$0")/.."
PORT="${1:-8888}"
WEEK="${WEEK:-week02-text-classification}"

# Only the lesson pages and their assets are served as web pages, never the whole repository (it may hold your .env file).
# They are copied fresh on every start, so after a "git pull" just start this script again.
SITE=".lesson-site"
rm -rf "$SITE"
mkdir -p "$SITE"
cp -r lessons assets "$SITE/"

# No "Jupyter news" pop-up over the notebook.
jupyter labextension disable "@jupyterlab/apputils-extension:announcements" >/dev/null 2>&1 || true

TOKEN="$(python -c 'import secrets; print(secrets.token_hex(16))')"
if [ -n "$CODESPACE_NAME" ]; then
  HOST="https://${CODESPACE_NAME}-${PORT}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
else
  HOST="http://localhost:${PORT}"
fi
LINK="${HOST}/login?token=${TOKEN}&next=%2Fstatic%2Flessons%2F${WEEK}.html"

echo
echo "=================================================================================="
echo " Lesson + notebook. Open this link (Ctrl+click, or Cmd+click on a Mac):"
echo
echo "   ${LINK}"
echo
echo " Keep this terminal open. Press Ctrl+C here to stop."
echo "=================================================================================="
echo

exec jupyter lab --no-browser --port "$PORT" --ServerApp.port_retries=0 \
  --IdentityProvider.token="$TOKEN" \
  --ServerApp.root_dir="$PWD" \
  --ServerApp.extra_static_paths="$PWD/$SITE" \
  --LabApp.expose_app_in_browser=True \
  --ServerApp.allow_remote_access=True \
  --ServerApp.allow_origin_pat='https://.*\.(app\.github\.dev|githubpreview\.dev)' \
  "${@:2}"
