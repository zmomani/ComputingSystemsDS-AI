#!/usr/bin/env bash
# The course notebook: lessons and their Jupyter notebooks side by side.
#
# Starts a Jupyter server in this codespace (in the background) and serves the lesson pages from it, so a lesson
# and its notebook share one browser window: the lesson on the left, the real notebook (PyTorch, Transformers, …)
# in the side panel. The button on each code example then takes you to the notebook cell that holds it.
#
#   bash scripts/start-lesson-lab.sh          start it (or, if it is already running, show the link again)
#   bash scripts/start-lesson-lab.sh stop     stop it
#
# The codespace runs this by itself every time you open it.
cd "$(dirname "$0")/.." || exit 1
PORT="${LESSON_LAB_PORT:-8888}"
STATE="$HOME/.lesson-lab"
mkdir -p "$STATE"

listening() { python -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 1)"; }

if [ "${1:-start}" = "stop" ]; then
  [ -f "$STATE/pid" ] && kill "$(cat "$STATE/pid")" 2>/dev/null
  rm -f "$STATE/pid"
  echo "The course notebook server is stopped."
  exit 0
fi

# One access token per codespace, kept outside the folder that is served as web pages.
[ -s "$STATE/token" ] || python -c 'import secrets; print(secrets.token_hex(16))' > "$STATE/token"
TOKEN="$(cat "$STATE/token")"

# Only the lesson pages and their assets are served as web pages, never the whole repository (it may hold your .env file).
# They are copied fresh on every run, so after a "git pull" just run this script again.
SITE=".lesson-site"
rm -rf "$SITE"
mkdir -p "$SITE"
cp -r index.html lessons assets "$SITE/"

if ! listening; then
  # No "Jupyter news" pop-up over the notebook.
  python - <<'PY' 2>/dev/null || true
import json, pathlib
p = pathlib.Path.home() / ".jupyter" / "labconfig" / "page_config.json"
p.parent.mkdir(parents=True, exist_ok=True)
cfg = json.loads(p.read_text()) if p.exists() else {}
cfg.setdefault("disabledExtensions", {})["@jupyterlab/apputils-extension:announcements"] = True
p.write_text(json.dumps(cfg, indent=2))
PY
  ROOT_FLAG=""; [ "$(id -u)" = "0" ] && ROOT_FLAG="--allow-root"
  nohup python -m jupyterlab --no-browser --port "$PORT" --ServerApp.port_retries=0 $ROOT_FLAG \
    --IdentityProvider.token="$TOKEN" \
    --ServerApp.root_dir="$PWD" \
    --ServerApp.extra_static_paths="$PWD/$SITE" \
    --LabApp.expose_app_in_browser=True \
    --ServerApp.allow_remote_access=True \
    --ServerApp.allow_origin_pat='https://.*\.(app\.github\.dev|githubpreview\.dev)' \
    > "$STATE/jupyter.log" 2>&1 &
  echo $! > "$STATE/pid"
  for _ in $(seq 1 60); do listening && break; sleep 1; done
fi

if ! listening; then
  echo "The course notebook did not start. The last lines of its log ($STATE/jupyter.log):"
  tail -n 15 "$STATE/jupyter.log"
  exit 1
fi

if [ -n "${CODESPACE_NAME:-}" ]; then
  HOST="https://${CODESPACE_NAME}-${PORT}.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
else
  HOST="http://localhost:${PORT}"
fi
LINK="${HOST}/login?token=${TOKEN}&next=%2Fstatic%2Findex.html"

echo
echo "=================================================================================="
echo " The course notebook is running."
echo " Open the lessons with their notebooks (Ctrl+click, or Cmd+click on a Mac):"
echo
echo "   ${LINK}"
echo
echo " Show this link again:  bash scripts/start-lesson-lab.sh"
echo " Stop the notebook:     bash scripts/start-lesson-lab.sh stop"
echo "=================================================================================="
echo

# In a codespace, also open the link in a new browser tab (the browser may ask for permission first).
if [ -n "${CODESPACE_NAME:-}" ] && [ -n "${BROWSER:-}" ]; then "$BROWSER" "$LINK" >/dev/null 2>&1 || true; fi
exit 0
