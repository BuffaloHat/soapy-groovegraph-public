# Essentia Extraction — Runbook

Purpose
- Extract per‑track audio features locally via Docker and write JSON outputs for later flattening.

Overview
- Essentia runbook: prerequisites, one-liner with caffeinate, what the runner does, common usage, progress/resume, troubleshooting, and next steps.

Prereqs
- Docker Desktop running (Apple Silicon OK; image runs as amd64 under emulation).
- Image auto‑pulled: `mtgupf/essentia:latest`.
- Music folder path (read‑only). Example: `/Users/mattkennedy/Music/Media.localized/Music`.

One‑liner (all files, resume‑safe)
- Keep the Mac awake and process with 2 jobs:
  - `caffeinate -dimsu python3 scripts/es_run_extractor.py --music-root "/Users/mattkennedy/Music/Media.localized/Music" --out-dir "/Users/mattkennedy/Projects/sgg/data/raw/essentia/extraction" --jobs 2`

What the runner does
- Scans recursively for `mp3/m4a/flac/wav/aiff/aif`; skips `.m4p` (DRM).
- Writes manifests for reproducibility:
  - `data/file_lists/es_all_rel.txt` (relative to music root)
  - `data/file_lists/es_all_abs.txt` (absolute paths)
- Runs the extractor via Docker per file and resumes by skipping existing non‑empty outputs.
- Outputs JSONs to `data/raw/essentia/extraction` as `<basename>-<shortHash>.json`.

Common usage
- Dry run 25 files: `python3 scripts/es_run_extractor.py --music-root "/Users/.../Music" --out-dir data/raw/essentia/extraction --limit 25 --dry-run`
- Full run (sequential): `python3 scripts/es_run_extractor.py --music-root "/Users/.../Music" --out-dir data/raw/essentia/extraction`
- Light parallelism (2–3 jobs): `python3 scripts/es_run_extractor.py --music-root "/Users/.../Music" --out-dir data/raw/essentia/extraction --jobs 2`
- Chunk: `python3 scripts/es_run_extractor.py --music-root "/Users/.../Music" --out-dir data/raw/essentia/extraction --start 0 --limit 500`

Progress & resume
- Planned total: `wc -l data/file_lists/es_all_rel.txt`
- Produced so far: `ls -1 data/raw/essentia/extraction | wc -l`
- Re‑run the same command to resume; existing outputs are skipped.

Troubleshooting
- DRM files: `.m4p` are protected and skipped.
- Permissions: grant Full Disk Access to Terminal/VS Code if reads fail.
- Performance: reduce `--jobs` or use smaller chunks.
- Fail logs: a `.log` appears next to the expected JSON when a file fails.

Next steps
- Flatten JSONs to Parquet and add dbt views (mirror the MB flattening flow).

---

# Flattening to Parquet — Runbook

Purpose
- Normalize Essentia JSON outputs into an analytics‑friendly Parquet dataset for DuckDB/dbt and vector prep.

Prereqs
- Activate the repo venv and install deps (includes `pandas` + `pyarrow`):
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`

Default (dataset) run
- Write a Parquet dataset into a folder of part files (append‑friendly):
  - `python scripts/es_flatten_features.py --overwrite`
- Output: `data/raw/essentia/essentia_features_v1/part-*.parquet`
- Notes:
  - Idempotent: `--overwrite` clears previous part files before writing.
  - Chunking: `--chunk-rows 5000` controls rows per part; increase to create fewer, larger parts.

Alternative: single file
- If you prefer one Parquet file for BI tools:
  - `python scripts/es_flatten_features.py --overwrite --out-parquet data/raw/essentia/essentia_features_v1.parquet`

What the flattener does
- Scalarizes tag arrays to strings for BI/dbt friendliness: `artist, album, title, date, albumartist, composer, file_name, genre`.
- Adds `genre_csv` (comma‑joined list) when multiple genres exist.
- Keeps true vectors as arrays: `mfcc_mean` (13), `thpcp` (36).
- Emits numeric features as scalars: `bpm, danceability, onset_rate, beats_count, loudness_integrated, loudness_range, spectral_*`, etc.
- Key detection: uses the first available among `tonal.key_edma` / `key_krumhansl` / `key_temperley` → `key_key, key_scale, key_strength`.
- Placeholders included (nullable for now): `voice_instrumental_*`, `mood_happy/mood_relaxed/mood_party`.

Quick validation
- DuckDB (dataset glob):
  - `duckdb -c "SELECT COUNT(*) FROM read_parquet('data/raw/essentia/essentia_features_v1/*.parquet');"`
  - `duckdb -c "SELECT artist, album, title, bpm, danceability, key_key, key_scale, loudness_integrated FROM read_parquet('data/raw/essentia/essentia_features_v1/*.parquet') LIMIT 10;"`
- Python (optional):
  - `python - <<'PY'\nimport pyarrow.dataset as ds; d=ds.dataset('data/raw/essentia/essentia_features_v1', format='parquet'); print('rows=', d.count_rows()); print(d.schema)\nPY`

Schema highlights (v1)
- Identity: `artist, album, title, date, genre, genre_csv, albumartist, composer, tracknumber, discnumber, file_name`.
- Audio props: `duration_sec, sample_rate, bit_rate, channels, codec, lossless, replay_gain`.
- Rhythm: `bpm, danceability, onset_rate, beats_count, bpm_peak1, bpm_peak2`.
- Tonal: `key_key, key_scale, key_strength, tuning_frequency, equal_tempered_deviation, chords_changes_rate`.
- Loudness/dynamics: `loudness_integrated, loudness_range, avg_loudness, dyn_complexity`.
- Spectral/timbral: `spectral_rms, spectral_centroid, spectral_flux, spectral_spread, pitch_salience, hfc, zcr`.
- Vectors: `mfcc_mean[13]`, `thpcp[36]`.
- Provenance: `features_version, essentia_version, extracted_with, file_hash, json_path, extracted_at`.

Vector hint (Qdrant)
- `sgg_audio_v1` vector order: `[danceability, bpm, loudness_integrated, loudness_range, spectral_rms, spectral_centroid, spectral_flux, pitch_salience, hfc, zcr, onset_rate, key_strength]`.
  - Keep `key_key`/`key_scale` as payload filters; add moods later in `v2`.

Troubleshooting
- `ModuleNotFoundError: pandas/pyarrow` → run `pip install -r requirements.txt` in the venv.
- DBeaver shows `VARCHAR[]` for tags → re‑run the updated flattener; tag fields are scalarized in v1.
- No mood columns populated → expected; requires enabling Essentia “highlevel” models and re‑extracting.
- Part files vs single file → use `--out-parquet` for one file; otherwise adjust `--chunk-rows`.
