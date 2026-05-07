# SGG — Infra, Ports & Config

> Source of truth: `infra_sgg/docker-compose.yml` + root `.env`. Never hardcode ports or URLs in code.

---

## Services & Ports

| Service | Purpose | Port(s) (host → container) | Env keys |
|---|---|---|---|
| **Qdrant** | Vector DB (REST/gRPC) | **6335→6333 / 6336→6334** | `QDRANT_URL`, `QDRANT_GRPC_URL`, `QDRANT_API_KEY` |
| **Ollama** | Local LLM + embeddings | **11434** | `OLLAMA_BASE_URL_HOST`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL` |
| **Postgres** | Future mart materialization target | **5433→5432** | `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD` |
| **Streamlit** | SGG dashboard UI | **TBD** | `STREAMLIT_PORT` |

> **Postgres** is running in Docker but not used in the active pipeline. It is reserved for future materialization of core Essentia marts. No changes to docker-compose needed.

---

## Service Details

- **Qdrant REST**: `http://localhost:6335` — always use the host port (6335), never 6333
- **Qdrant Dashboard**: `http://localhost:6335/dashboard`
- **Qdrant gRPC**: `localhost:6336`
- **Ollama**: `http://localhost:11434` — start with `ollama serve`; runs outside Docker on host
- **Postgres**: `localhost:5433`

---

## Health Checks

```bash
curl http://localhost:6335/healthz          # Qdrant
curl http://localhost:11434/api/tags        # Ollama — lists pulled models
psql -h localhost -p 5433 -U sgg -c "SELECT now();"   # Postgres
```

---

## Ollama Models

| Role | Model | Env key |
|---|---|---|
| Chat / RAG answer | `gemma3:27b` | `OLLAMA_CHAT_MODEL` |
| Text embeddings | `nomic-embed-text` | `OLLAMA_EMBED_MODEL` |
| Reasoning (reserved) | `qwen2.5:7b` | `OLLAMA_REASON_MODEL` |

---

## Core Config Keys (`/.env`)

```ini
# Qdrant
QDRANT_URL=http://localhost:6335
QDRANT_API_KEY=

# Ollama
OLLAMA_BASE_URL_HOST=http://localhost:11434
OLLAMA_BASE_URL_FROM_DOCKER=http://host.docker.internal:11434
OLLAMA_CHAT_MODEL=gemma3:27b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_REASON_MODEL=qwen2.5:7b
EMBEDDINGS_BACKEND=ollama

# Postgres
PG_HOST=localhost
PG_PORT=5432
PG_DB=sgg
PG_USER=sgg
PG_PASSWORD=sgg

# Paths
DATA_ROOT=./data
RUNS_DIR=./runs
```

---

## Essentia (Docker)

Used for one-time or periodic audio feature extraction from the iTunes library. Not part of the ongoing query pipeline.

- **Image**: `mtgupf/essentia:latest`
- **Source audio**: iTunes library (read-only bind mount)
- **Outputs**: JSON per track → `data/raw/essentia/extraction/`
- **DRM**: `.m4p` files are protected — skip them (handled automatically by extraction scripts)

Run extraction:
```bash
python scripts/es_run_extractor.py \
  --music-root "/Users/<you>/Music/Media.localized/Music" \
  --out-dir data/raw/essentia/extraction \
  --jobs 2
```

Keep Mac awake during long runs:
```bash
caffeinate -dimsu python scripts/es_run_extractor.py ...
```

One-off test:
```bash
docker run --rm --platform=linux/amd64 \
  -v "<music_root>:/in:ro" \
  -v "$PWD/data/raw/essentia/extraction:/out" \
  mtgupf/essentia:latest \
  /usr/local/bin/essentia_streaming_extractor_music "/in/<Artist>/<Album>/<Track>.mp3" "/out/test.json"
```

---

## iTunes Search API

Used by the image pipeline to resolve album artwork URLs. No API key required.

- **Endpoint**: `https://itunes.apple.com/search?term=<artist+album>&entity=album&limit=1`
- **Returns**: `artworkUrl100` (replace `100` with `600` for full resolution)
- **Cache**: Results saved to `data/cache/itunes_image_cache.json` — always check cache before calling the API
- **Rate limiting**: Undocumented — cache aggressively; batch lookups with small delays
