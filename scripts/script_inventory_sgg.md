# SGG Scripts Inventory
**Last updated:** 2026-04-20

Reference for all active scripts in `scripts/`. Each entry describes the purpose, inputs/outputs, CLI usage, and when to re-run. Archived scripts (MusicBrainz, SRP-related) are in `scripts/z_archive/`.

---

## Pipeline Overview

The scripts follow a linear pipeline from raw iTunes audio to a queryable RAG system:

```
iTunes Library
    │
    ├─ es_select_files.py       → build file manifest
    ├─ es_run_extractor.py      → Essentia Docker extraction → JSON files
    ├─ es_flatten_features.py   → JSON → Parquet
    │       (then: dbt run)
    ├─ qdrant_upsert_audio.py   → Parquet → sgg_audio_v1 (numeric vectors)
    ├─ sgg_text_embed.py        → Parquet → sgg_text_v1 (text embeddings) + RAG
    ├─ itunes_image_lookup.py   → build iTunes album art cache
    └─ eval_audio_similarity.py → spot-check audio similarity results
```

---

## `es_select_files.py`

**Purpose:** Build a manifest of iTunes audio files to pass to the Essentia extractor. Scans the Apple Music library directory and outputs lists of file paths. Useful for targeted extractions (by artist prefix or subset) before doing a full library run.

| | |
|---|---|
| **Inputs** | `--music-root` — path to Apple Music library root |
| **Outputs** | `data/file_lists/es_targets.txt` (relative paths), `es_targets_abs.txt` (absolute), `es_match_report.csv` (match audit) |
| **When to re-run** | When you add music or want a targeted subset for extraction |

```bash
# Full library manifest
python scripts/es_select_files.py \
  --music-root "/Users/<you>/Music/Media.localized/Music"

# Filter to a single artist prefix
python scripts/es_select_files.py \
  --music-root "/Users/<you>/Music/Media.localized/Music" \
  --artist-prefix "B" --lenient
```

---

## `es_run_extractor.py`

**Purpose:** Batch-run the Essentia streaming extractor via Docker over the iTunes library. Resume-safe — skips files already extracted. Outputs one JSON per track containing all low-level audio features and iTunes tag metadata.

| | |
|---|---|
| **Prereqs** | Docker Desktop running; `mtgupf/essentia:latest` (auto-pulled on first run) |
| **Inputs** | `--music-root` — scanned recursively; `.m4p` DRM files are automatically skipped |
| **Outputs** | `data/raw/essentia/extraction/<trackname>-<hash>.json` — one file per track |
| **When to re-run** | When you add new music to your iTunes library |

```bash
# Dry run — see what would be processed
python scripts/es_run_extractor.py \
  --music-root "/Users/<you>/Music/Media.localized/Music" \
  --out-dir data/raw/essentia/extraction \
  --limit 25 --dry-run

# Full extraction (2 parallel jobs)
python scripts/es_run_extractor.py \
  --music-root "/Users/<you>/Music/Media.localized/Music" \
  --out-dir data/raw/essentia/extraction \
  --jobs 2

# Keep Mac awake during long runs
caffeinate -dimsu python scripts/es_run_extractor.py ...
```

---

## `es_flatten_features.py`

**Purpose:** Flatten the Essentia JSON extraction outputs into a structured Parquet dataset for dbt ingestion. Scalarizes tag arrays (artist, album, title, genre), preserves numeric vectors (MFCC, THPCP) as arrays, and adds a `file_hash` primary key.

| | |
|---|---|
| **Inputs** | `data/raw/essentia/extraction/*.json` |
| **Outputs** | `data/raw/essentia/essentia_features_v1/part-*.parquet` (partitioned dataset) |
| **When to re-run** | After a new Essentia extraction run; run `dbt run` afterward to refresh views |

```bash
# Full flatten (overwrites existing Parquet)
python scripts/es_flatten_features.py --overwrite

# With optional single-file output
python scripts/es_flatten_features.py --overwrite \
  --out-parquet data/raw/essentia/essentia_features_v1.parquet

# Test on a subset
python scripts/es_flatten_features.py --limit 500 \
  --out-parquet data/raw/essentia/essentia_features_sample.parquet
```

---

## `qdrant_upsert_audio.py`

**Purpose:** Manage the `sgg_audio_v1` Qdrant collection. Handles collection creation, computing and freezing z-score stats, upserting z-scored 12-dim audio vectors, and querying top-k neighbors by `file_hash`. This is the audio similarity search backend.

| | |
|---|---|
| **Source** | `fct_audio_vector_v1` view in `dbt_sgg/sgg_prod.duckdb` |
| **Collection** | `sgg_audio_v1` — dim=12, cosine similarity |
| **Stats file** | `data/features/sgg_audio_v1_stats.json` — frozen z-score parameters |
| **Env** | `QDRANT_URL` (default: `http://localhost:6335`), `QDRANT_API_KEY` (optional) |
| **When to re-run** | After adding new tracks (re-run `stats` → `upsert`); `init` only needed once |

```bash
# One-time setup — create collection and payload indexes
python scripts/qdrant_upsert_audio.py init

# Compute and freeze z-score stats
python scripts/qdrant_upsert_audio.py stats \
  --out-json data/features/sgg_audio_v1_stats.json

# Upsert all tracks with z-scored vectors
python scripts/qdrant_upsert_audio.py upsert \
  --vector z \
  --stats-json data/features/sgg_audio_v1_stats.json \
  --batch-size 1000

# Query top-10 neighbors for a track
python scripts/qdrant_upsert_audio.py query \
  --file-hash <hash> --topk 10

# Query with filters
python scripts/qdrant_upsert_audio.py query \
  --file-hash <hash> --topk 10 --genre Jazz --key_key C
```

---

## `sgg_text_embed.py`

**Purpose:** Manage the `sgg_text_v1` Qdrant collection and run the RAG answer pipeline. Each track is converted to a natural-language sentence and embedded via Ollama (`nomic-embed-text`). The `rag` subcommand retrieves top-k semantic neighbors and generates a cited answer via `gemma3:27b`.

| | |
|---|---|
| **Source** | `fct_audio_vector_v1` view in `dbt_sgg/sgg_prod.duckdb` |
| **Collection** | `sgg_text_v1` — dim=768, cosine similarity |
| **Embed model** | `nomic-embed-text` via Ollama |
| **Chat model** | `gemma3:27b` via Ollama |
| **Env** | `QDRANT_URL`, `OLLAMA_BASE_URL_HOST`, `OLLAMA_EMBED_MODEL`, `OLLAMA_CHAT_MODEL` |
| **When to re-run** | `embed` after adding new tracks; `init` only needed once |

**Document format per track:**
```
"<Title> by <Artist>, from the album <Album> (<Year>).
 Genre: <Genre>. Musical key: <Key> <Scale>. BPM: <BPM>.
 Energy: <label>. Danceability: <label>."
```

```bash
# One-time setup — create collection
python scripts/sgg_text_embed.py init

# Embed all tracks and upsert
python scripts/sgg_text_embed.py embed

# Test on a subset first
python scripts/sgg_text_embed.py embed --limit 100

# Semantic search (returns top-k results)
python scripts/sgg_text_embed.py query \
  --text "slow jazzy late night music" --topk 10

# RAG — full answer with citations
python scripts/sgg_text_embed.py rag \
  --text "upbeat energetic music to wake up to" --topk 10
```

---

## `itunes_image_lookup.py`

**Purpose:** Build the album artwork URL cache by querying the iTunes Search API for every unique artist+album pair in the Essentia dataset. Results are cached locally so the Streamlit app never calls the API at query time. Resume-safe — skips already-resolved pairs and retries `RETRY` entries from previous rate-limited runs.

| | |
|---|---|
| **Source** | `im_essentia_features_unique` in `dbt_sgg/sgg_prod.duckdb` |
| **Outputs** | `data/cache/itunes_image_cache.json` (lookup dict), `data/cache/itunes_image_cache.parquet` (flat table) |
| **API** | iTunes Search API — no key required |
| **Pacing** | 2s between requests; pauses 90s every 50 requests to avoid rate limiting |
| **Cache values** | URL string = hit, `null` = genuine miss, `"RETRY"` = rate-limited (will retry on next run) |
| **When to re-run** | After adding new music (new artist+album pairs); or to retry `RETRY` entries |

```bash
# Run / resume full lookup (safe to re-run)
python scripts/itunes_image_lookup.py

# Check hit rate without running
python scripts/itunes_image_lookup.py --stats

# Test on first 30 pairs only
python scripts/itunes_image_lookup.py --limit 30
```

---

## `eval_audio_similarity.py`

**Purpose:** Export an evaluation CSV from `sgg_audio_v1` for spot-checking audio similarity quality. For each seed track, retrieves the top-k nearest neighbors and writes a flat CSV with seed identity, neighbor identity, similarity score, and key audio features for manual review.

| | |
|---|---|
| **Source** | `sgg_audio_v1` Qdrant collection + `fct_audio_vector_v1` DuckDB view |
| **Outputs** | `data/eval/audio_similarity_eval_<year>.csv` |
| **When to re-run** | After upserting new vectors to verify similarity quality |

```bash
# Run eval with default seeds
python scripts/eval_audio_similarity.py \
  --out data/eval/audio_similarity_eval_2026.csv \
  --topk 10

# Custom seed file hashes
python scripts/eval_audio_similarity.py \
  --seeds <hash1> <hash2> <hash3> \
  --out data/eval/audio_similarity_eval_2026.csv
```

---

## `start_sgg.sh`

**Purpose:** Shell script that brings up the full SGG stack in one command. Called by `make dashboard`. Starts Docker services (Qdrant + Postgres), checks if Ollama is already running and starts it if not, then launches the Streamlit dashboard.

| | |
|---|---|
| **Prereqs** | Docker Desktop running; venv installed; Ollama installed |
| **Outputs** | Streamlit dashboard at `http://localhost:8501` |
| **When to re-run** | Any time you want to start the full stack from scratch |

```bash
# Via Makefile (preferred)
make dashboard

# Direct
bash scripts/start_sgg.sh
```

---

## General Notes

- Always run from repo root (not from inside `scripts/`)
- Activate venv first: `source venv/bin/activate`
- All scripts load `.env` via `python-dotenv` — ensure services are running before executing Qdrant or Ollama scripts
- Health checks before running: `curl http://localhost:6335/healthz` (Qdrant), `curl http://localhost:11434/api/tags` (Ollama)
- Long-running extractions: use `caffeinate -dimsu <command>` on macOS to prevent sleep
- Archived scripts (MusicBrainz, SRP): `scripts/z_archive/`
