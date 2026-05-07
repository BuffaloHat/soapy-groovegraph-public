# Soapy GrooveGraph (SGG) — Overview
**Last updated:** 2026-04-20

---

## 1. What SGG Is

**Soapy GrooveGraph (SGG)** is an AI-enhanced music discovery layer over a personal iTunes audio library. Users ask natural language questions and receive LLM-generated, cited answers — powered by Essentia audio features and Qdrant vector search.

SGG is a *learning-first* environment focused on building durable skills in audio feature engineering, vector databases, embeddings, and retrieval-augmented generation (RAG) using open, local-first technologies.

> Note: SGG is a standalone AI/ML project. A separate Discogs/Metabase project (Soapy Records Project) handles traditional collection browsing and querying.

---

## 2. Core Concept

> "Soapy GrooveGraph learns the language of your music"

The system extracts numeric and semantic features from a personal iTunes library using **Essentia**, stores them as vectors in **Qdrant**, and answers natural language queries via a RAG pipeline backed by **Ollama**.

### Example Questions SGG Answers
- "What are a few slow jazzy late night tracks?"
- "Give me upbeat energetic music to wake up to."
- "Which tracks share similar tempo and mood to this one?"
- "Recommend something melancholy and slow for a rainy day."

---

## 3. Tool Stack

| **Layer** | **Tool / Service** | **Purpose** |
|---|---|---|
| Audio Feature Extraction | **Essentia (Docker)** | Extract BPM, key, danceability, mood, energy from iTunes audio files |
| Data Transformation | **dbt-core + DuckDB** | Stage, deduplicate, and build the audio vector mart (`fct_audio_vector_v1`) |
| Data Inspection | **DBeaver** | Browse Parquet files and DuckDB views during development |
| Embeddings | **Ollama — nomic-embed-text** | Generate 768-dim text embeddings from track-level natural language sentences |
| Vector Store | **Qdrant** | Store audio feature vectors (`sgg_audio_v1`) and text embeddings (`sgg_text_v1`) |
| Local LLM | **Ollama — gemma3:27b** | Generate cited natural language answers from retrieved context |
| Album Art | **iTunes Search API (free)** | Resolve album artwork URLs for visual UI enrichment — no API key required |
| Frontend | **Streamlit** | Interactive UI: RAG query panel, audio similarity search, feature inspector |
| Version Control | **GitHub / Markdown** | Track iterations, architecture notes, and session progress |

---

## 4. Data Flow Architecture

```
iTunes Audio Library
       │
       ▼
Essentia (Docker) — audio feature extraction
       │
       ▼
data/raw/essentia/extraction/*.json
       │
       ▼
dbt-core + DuckDB
  stg_essentia_features
  im_essentia_features_unique
  fct_audio_vector_v1
       │
       ├──► Qdrant: sgg_audio_v1 (12-dim numeric z-score vectors)
       │
       └──► Ollama (nomic-embed-text)
                   │
                   ▼
            Qdrant: sgg_text_v1 (768-dim text embeddings)
                   │
                   ▼
           RAG Layer (Ollama — gemma3:27b)
                   │
                   ▼
         Streamlit UI + iTunes album art
```

---

## 5. Data Sources

### Essentia — Primary and Only Source
- **What it is**: Audio features extracted from personal iTunes library using the Essentia Docker image
- **Coverage**: 9,294 tracks, 558 artists, 841 albums
- **Key fields**: `file_hash` (primary key), `artist`, `album`, `title`, `date`, `genre`, `bpm`, `key_key`, `key_scale`, `key_strength`, `danceability`, `loudness_integrated`, `mood_happy`, `mood_relaxed`, `mood_party`, plus z-scored vector versions
- **Qdrant collections**: `sgg_audio_v1` (dim=12, cosine) and `sgg_text_v1` (dim=768, cosine)
- **Known gaps**: ~483 tracks missing date, 18 missing title — minor, acceptable for this use case

### iTunes Search API — Image Enrichment Only
- **What it is**: Free Apple API used to resolve album artwork URLs at query time
- **Usage**: Artist + album name → `artworkUrl600`; results cached locally in `data/cache/itunes_image_cache.json`
- **Not used for**: metadata enrichment (Essentia iTunes tags are sufficient)

### Archived Sources (not in active pipeline)
- **SRP / Discogs**: 243 album records from personal Discogs collection — covered by the separate Soapy Records Project (Metabase); archived in `archive/`
- **MusicBrainz**: 302K release groups fetched for 150 SRP artists — made redundant by Essentia iTunes tags; archived in `archive/`

---

## 6. Folder Structure

```
sgg/
├── ai_sgg/                  # Python package
│   ├── embedding/           # Embedding generation and search
│   ├── features/            # Audio feature utilities
│   ├── ingest/              # Data ingestion helpers
│   └── utils/               # Shared config, logging, iTunes image lookup
│
├── apps/                    # Streamlit UI
│   └── sgg_dashboard.py
│
├── archive/                 # Preserved out-of-scope code and data
│   ├── scripts/             # MB and SRP scripts
│   └── dbt/                 # MB and SRP dbt models
│
├── data/
│   ├── raw/essentia/        # Source of truth — Essentia JSON outputs and Parquet
│   ├── features/            # Qdrant stats JSON, feature manifests
│   ├── eval/                # Audio similarity eval CSVs
│   ├── cache/               # iTunes image URL cache (JSON + Parquet)
│   └── file_lists/          # iTunes library manifests for Essentia extraction
│
├── dbt_sgg/                 # dbt project (DuckDB dev profile → sgg_prod.duckdb)
│   └── models/
│       ├── staging/         # stg_essentia_features, ext_essentia_features
│       ├── intermediate/    # im_essentia_features_unique
│       └── marts/           # fct_audio_vector_v1
│
├── docs/                    # Project documentation
│   ├── overview.md          # This document
│   ├── todo.md              # Done / Next / Parking Lot task list
│   ├── progress.md          # Session history and next steps
│   ├── infra.md             # Ports, services, and config reference
│   ├── data_inventory.md    # Data source field inventory and gap analysis
│   └── archive/             # Deprecated and superseded docs
│
├── infra_sgg/               # Docker Compose stack
│   └── docker-compose.yml   # Qdrant, Postgres, Ollama (host)
│
├── notebooks/               # Jupyter development notebooks
│   └── text_rag_dev.ipynb
│
├── prompts/                 # RAG prompt templates
├── runs/                    # Run manifests (model names, counts, hashes)
├── scripts/                 # CLI entry points (argparse)
├── .env                     # Single source of truth for all env config
├── CLAUDE.md                # Claude Code guidance for this repo
├── Makefile                 # make up / down / ps / help
└── requirements.txt
```

---

## 7. AI Learning Objectives

- Build and validate end-to-end RAG pipelines grounded in both numeric and text vectors
- Learn audio feature engineering with Essentia — BPM, key, energy, mood classifiers
- Develop hands-on fluency with Qdrant (collections, upserts, cosine similarity, payloads)
- Practice prompt engineering for grounded, cited LLM answers using local models (Ollama)
- Build a functional Streamlit UI integrating vector search and LLM generation
- Maintain full local-first control — no commercial APIs required for core functionality
- Apply dbt modeling discipline (staging → intermediate → marts) to non-traditional data sources
