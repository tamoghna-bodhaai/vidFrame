#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  uv venv --python /usr/bin/python3 --system-site-packages .venv
fi

if ! .venv/bin/python -c "import uvicorn, cv2, fastapi, httpx, manim" >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements.txt
fi

if [ ! -f .pdf-render/node_modules/katex/dist/katex.min.js ]; then
  npm install --prefix .pdf-render
fi

exec .venv/bin/python -m uvicorn app.main:app \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  "$@"
