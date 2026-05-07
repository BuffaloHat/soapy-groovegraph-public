#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Starting Docker services (Qdrant + Postgres)..."
docker compose -f infra_sgg/docker-compose.yml up -d

echo "==> Starting Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "    Ollama already running — skipping."
else
    ollama serve &
    OLLAMA_PID=$!
    echo "    Ollama started (pid $OLLAMA_PID)"
    sleep 2
fi

echo "==> Starting Streamlit dashboard..."
source "$REPO_ROOT/venv/bin/activate"
streamlit run apps/sgg_dashboard.py
