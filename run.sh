#!/bin/bash
# Launch Clip Reels.
#   ./run.sh
# To enable AI moment-selection, set your key first:
#   export ANTHROPIC_API_KEY=sk-ant-...
cd "$(dirname "$0")"

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "⚠️  ANTHROPIC_API_KEY not set — the app will use the fallback selector."
  echo "   To enable AI: export ANTHROPIC_API_KEY=sk-ant-...   then re-run."
  echo
fi

./venv/bin/python app.py
