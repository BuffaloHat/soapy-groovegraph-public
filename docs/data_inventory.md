# SGG — Data Inventory
**Last updated:** 2026-04-20

Technical reference for all data sources in the SGG pipeline — field schemas, row counts, known gaps, Qdrant collection specs, and join strategy. Essentia (audio feature extraction from a personal iTunes library) is the sole active data source for v1.

Complete inventory of data sources, fields, dimensions, metrics, known gaps, and join strategy for the current SGG pipeline.

---

## Summary

| Source | Status | Records | Key | Purpose |
|---|---|---|---|---|
| **Essentia** | Active — primary source | 9,294 tracks | `file_hash` | Audio features + iTunes metadata |
| **Qdrant `sgg_audio_v1`** | Active | 9,294 points | `file_hash` | 12-dim numeric similarity search |
| **Qdrant `sgg_text_v1`** | Active | 9,294 points | `file_hash` | 768-dim semantic/RAG search |
| **iTunes image cache** | Active — building | 999 artist/album pairs | `artist\|album` | Album artwork for UI |
| **SRP / Discogs** | Archived | 243 albums | `release_id` | Covered by separate SRP project |
| **MusicBrainz** | Archived | 302K release groups | `artist_mbid` | Redundant — Essentia has iTunes tags |

---

## 1. Essentia — Primary Data Source

### Overview
Audio features extracted from a personal iTunes library using the Essentia Docker image (`mtgupf/essentia:latest`). Each track produces one JSON file; all JSONs are flattened to Parquet and loaded into dbt.

- **Source files**: `data/raw/essentia/extraction/*.json` (9,294+ files)
- **Flattened Parquet**: `data/raw/essentia/essentia_features_v1/part-*.parquet`
- **Primary key**: `file_hash` (SHA-based hash of the source file path)
- **dbt entry point**: `ext_essentia_features` → `stg_essentia_features` → `im_essentia_features_unique` → `fct_audio_vector_v1`

### Coverage
| Dimension | Value |
|---|---|
| Total tracks | 9,294 |
| Unique artists | 558 |
| Unique albums | 841 |
| Average track duration | 4.8 min |
| BPM range | 65.4 – 191.4 |
| Date range (non-null) | 1928 – 2017 |
| Audio formats | MP3 (7,758), AAC (1,379), ALAC lossless (157) |

### iTunes Tag Metadata Fields
Sourced from embedded ID3/AAC tags at extraction time.

| Field | Type | Notes |
|---|---|---|
| `file_hash` | VARCHAR | Primary key |
| `artist` | VARCHAR | Track artist (iTunes tag) |
| `album` | VARCHAR | Album name (iTunes tag) |
| `title` | VARCHAR | Track title (iTunes tag) |
| `date` | VARCHAR | Release date string (year or full date) |
| `genre` | VARCHAR | iTunes genre tag — see distribution below |
| `genre_csv` | VARCHAR | Raw genre tag, comma-separated if multiple |
| `albumartist` | VARCHAR | Album-level artist (may differ from track artist) |
| `composer` | VARCHAR | Composer (populated for classical) |
| `tracknumber` | INTEGER | Track position on album |
| `discnumber` | INTEGER | Disc number for multi-disc sets |
| `file_name` | VARCHAR | Source filename |

### Audio Format Fields
| Field | Type | Notes |
|---|---|---|
| `duration_sec` | DOUBLE | Track length in seconds |
| `duration_min` | DOUBLE | Track length in minutes |
| `sample_rate` | BIGINT | e.g. 44100, 48000 |
| `bit_rate` | BIGINT | Encoding bitrate |
| `channels` | BIGINT | 1=mono, 2=stereo |
| `codec` | VARCHAR | mp3float, aac, alac |
| `lossless` | BOOLEAN | True for ALAC only |
| `replay_gain` | DOUBLE | ReplayGain normalization value |

### Rhythm & Tempo Fields
| Field | Type | Notes |
|---|---|---|
| `bpm` | DOUBLE | Beats per minute — primary tempo signal |
| `bpm_peak1` | BIGINT | First BPM peak candidate |
| `bpm_peak2` | BIGINT | Second BPM peak candidate |
| `beats_count` | BIGINT | Total beat count |
| `danceability` | DOUBLE | Range: 0.714 – 2.508; higher = more danceable |
| `onset_rate` | DOUBLE | Note onset events per second |

### Key & Harmony Fields
| Field | Type | Notes |
|---|---|---|
| `key_key` | VARCHAR | Musical key: C, D, E, F, G, A, B, Bb, Eb, Ab |
| `key_scale` | VARCHAR | major or minor |
| `key_strength` | DOUBLE | Confidence of key detection (0–1) |
| `tuning_frequency` | DOUBLE | Concert pitch (typically ~440 Hz) |
| `equal_tempered_deviation` | DOUBLE | Deviation from equal temperament |
| `chords_changes_rate` | DOUBLE | Rate of chord changes |

### Loudness & Dynamics Fields
| Field | Type | Notes |
|---|---|---|
| `loudness_integrated` | DOUBLE | Integrated LUFS loudness — primary energy signal |
| `loudness_range` | DOUBLE | Dynamic range (LU) |
| `avg_loudness` | DOUBLE | Average loudness |
| `dyn_complexity` | DOUBLE | Dynamic complexity index |

### Spectral / Timbral Fields
| Field | Type | Notes |
|---|---|---|
| `spectral_rms` | DOUBLE | RMS energy — timbral brightness proxy |
| `spectral_centroid` | DOUBLE | Brightness of sound (higher = brighter) |
| `spectral_flux` | DOUBLE | Rate of spectral change — texture signal |
| `spectral_spread` | DOUBLE | Spectral width |
| `pitch_salience` | DOUBLE | Prominence of pitched content (0–1) |
| `hfc` | DOUBLE | High-frequency content |
| `zcr` | DOUBLE | Zero-crossing rate — noisiness signal |
| `mfcc_mean` | DOUBLE[13] | Mel-frequency cepstral coefficients (timbre) |
| `thpcp` | DOUBLE[36] | Tonal pitch class profile (harmonic content) |

### Mood & Voice Fields ⚠️
| Field | Type | Notes |
|---|---|---|
| `mood_happy` | DOUBLE | **All NULL** — high-level models not run at extraction |
| `mood_relaxed` | DOUBLE | **All NULL** — same |
| `mood_party` | DOUBLE | **All NULL** — same |
| `mood_model` | VARCHAR | **All NULL** |
| `voice_instrumental_value` | VARCHAR | **All NULL** |
| `voice_instrumental_prob_vocal` | DOUBLE | **All NULL** |
| `voice_instrumental_prob_instrumental` | DOUBLE | **All NULL** |

> These fields require Essentia's high-level SVM classifier models which were not included in the Docker extraction run. They are placeholders for a future `sgg_audio_v2` re-extraction. The RAG and similarity systems work without them.

### Extraction Metadata Fields
| Field | Type | Notes |
|---|---|---|
| `features_version` | VARCHAR | Essentia feature schema version |
| `essentia_version` | VARCHAR | Essentia library version |
| `extracted_with` | VARCHAR | Extractor binary name |
| `json_path` | VARCHAR | Source JSON file path |
| `extracted_at` | VARCHAR | Extraction timestamp |

### Known Data Gaps
| Issue | Count | Impact |
|---|---|---|
| Missing `date` | 483 tracks (5.2%) | Year shown as "unknown year" in text embeddings |
| Missing `title` | 18 tracks (0.2%) | Shown as "Unknown Title" in embeddings |
| Missing `genre` | 160 tracks (1.7%) | Shown as "Unknown Genre" in embeddings |
| Mood fields all NULL | 9,294 tracks (100%) | Not used in v1 pipeline; planned for v2 |
| Multi-disc albums | Various | `(Disc 1)` suffix affects iTunes image lookup |

### Genre Distribution (top 10)
| Genre | Tracks |
|---|---|
| Rock | 3,660 |
| Jazz | 1,047 |
| Alternative & Punk | 785 |
| Alternative | 636 |
| Country | 406 |
| R&B | 300 |
| Reggae | 294 |
| Pop | 278 |
| Blues | 232 |
| Hip Hop/Rap | 222 |

### Key Distribution (top 5)
| Key | Count |
|---|---|
| C major | 1,122 |
| G major | 1,088 |
| D major | 965 |
| A major | 922 |
| F major | 693 |

---

## 2. dbt Model Chain

All models use DuckDB dev profile (`dbt_sgg/sgg_prod.duckdb`).

| Model | Type | Rows | Description |
|---|---|---|---|
| `ext_essentia_features` | View (external) | 9,297 | DuckDB glob over Parquet files in `essentia_features_v1/` |
| `stg_essentia_features` | View | 9,297 | Typed staging — casts all columns, adds `duration_min` |
| `im_essentia_features_unique` | View | 9,294 | Deduped by `file_hash` (removes 3 duplicates) |
| `fct_audio_vector_v1` | View | 9,294 | 12-dim `vector_raw` and z-scored `vector_z` for Qdrant |

### `fct_audio_vector_v1` — Vector Columns
The 12 features used in `sgg_audio_v1` (in order):

| Position | Feature | Raw field |
|---|---|---|
| 1 | Danceability | `danceability` |
| 2 | BPM | `bpm` |
| 3 | Integrated loudness | `loudness_integrated` |
| 4 | Loudness range | `loudness_range` |
| 5 | Spectral RMS | `spectral_rms` |
| 6 | Spectral centroid | `spectral_centroid` |
| 7 | Spectral flux | `spectral_flux` |
| 8 | Pitch salience | `pitch_salience` |
| 9 | High-frequency content | `hfc` |
| 10 | Zero-crossing rate | `zcr` |
| 11 | Onset rate | `onset_rate` |
| 12 | Key strength | `key_strength` |

Z-score stats frozen in `data/features/sgg_audio_v1_stats.json` for reproducibility.

---

## 3. Qdrant Collections

### `sgg_audio_v1` — Numeric Audio Vectors
| Property | Value |
|---|---|
| Dimensions | 12 |
| Distance metric | Cosine |
| Point ID | `file_hash` |
| Point count | 9,294 |
| Vector type | Z-scored (`vector_z` from `fct_audio_vector_v1`) |

**Payload fields per point:** `artist`, `album`, `title`, `date`, `genre`, `key_key`, `key_scale`, `vector_profile`

**Use case:** Audio similarity search — "find tracks that sound like this one" by numeric feature proximity.

### `sgg_text_v1` — Text Embeddings
| Property | Value |
|---|---|
| Dimensions | 768 |
| Distance metric | Cosine |
| Point ID | `file_hash` |
| Point count | 9,294 |
| Embedding model | `nomic-embed-text` via Ollama |

**Payload fields per point:** `artist`, `album`, `title`, `date`, `genre`, `key`, `bpm`, `energy_label`, `danceability_label`

**Document format per track:**
```
"<Title> by <Artist>, from the album <Album> (<Year>).
 Genre: <Genre>. Musical key: <Key> <Scale>. BPM: <BPM>.
 Energy: <quiet and gentle|moderate energy|loud and energetic>.
 Danceability: <low|moderate|high>."
```

**Use case:** Semantic / natural language search — "slow jazzy late night music", "upbeat energetic wake-up tracks".

**Energy labels** (derived from `loudness_integrated`):
- `quiet and gentle` — below -16 LUFS
- `moderate energy` — -16 to -10 LUFS
- `loud and energetic` — above -10 LUFS

---

## 4. iTunes Image Cache

### `data/cache/itunes_image_cache.json`
| Property | Value |
|---|---|
| Source | iTunes Search API (no key required) |
| Key format | `"artist_lowercase\|album_lowercase"` |
| Value | Artwork URL string, `null` (genuine miss), or `"RETRY"` (pending) |
| Unique pairs | 999 |
| Hit rate | 75.8% (757 hits, 242 genuine misses) |

**URL format:** `https://is1-ssl.mzstatic.com/.../600x600bb.jpg`

**Cache is pre-built** — the Streamlit app loads it into memory at startup via `apps/itunes_image.py`. The API is never called at query time.

**Known miss patterns:**
- Multi-disc album suffixes: `(Disc 1)`, `[Disc 2]`
- Obscure/compilation-only artists
- Comedy albums
- Very long subtitle strings

### `data/cache/itunes_image_cache.parquet`
Flattened version of the JSON cache with columns: `cache_key`, `artist`, `album`, `artwork_url`. Used for DuckDB joins and dbt external views if needed.

---

## 5. Archived Data (preserved, not in active pipeline)

### `data/raw/seed/`
- **Contents**: `fct_music_inventory.csv` — 243 album records from Discogs collection (artist, album_name, release_year, track_count)
- **Why archived**: Covered by the separate Soapy Records Project (Metabase). No track-level data or audio features.
- **Reuse**: Available for SRP project; could identify "curated collection" subset of Essentia tracks in a future SGG iteration.

### `data/raw/musicbrainz/`
- **Contents**: Release group JSONs for 150 artists, artist resolution files, MBID mappings, flattened Parquet (302K release group records)
- **Why archived**: Intended to enrich Essentia with canonical metadata via MBID join key. Made redundant when Essentia was found to capture iTunes tags at extraction time. The MBID-to-Essentia join was never completed.
- **Reuse**: High value for SRP project — canonical MBIDs, release group data, Discogs/Wikidata cross-links.

---

## 6. Join Strategy

The active pipeline has **no cross-source joins**. Essentia is self-contained.

| Join | Status | Blocker |
|---|---|---|
| Essentia ↔ iTunes image cache | Active (by `artist\|album` key) | ~20% miss rate on album name matching |
| Essentia ↔ SRP/Discogs | Not built | No track-level data in SRP; fuzzy artist+album match needed |
| Essentia ↔ MusicBrainz | Not built | No MBID on Essentia tracks; would require fuzzy resolution |

---

## 7. Evaluation Data

### `data/eval/audio_similarity_eval_2026.csv`
40-row evaluation of `sgg_audio_v1` similarity results.

| Column | Description |
|---|---|
| `seed_hash` | `file_hash` of the query track |
| `seed_artist` / `seed_title` | Seed track identity |
| `neighbor_hash` | `file_hash` of the returned neighbor |
| `neighbor_artist` / `neighbor_title` | Neighbor track identity |
| `score` | Cosine similarity score |
| `genre` / `key_key` / `bpm` | Neighbor audio features for spot-check |

4 seeds tested: Pink Floyd (atmospheric rock), Wolf Parade (high-energy indie), Norah Jones (quiet acoustic pop), Thelonious Monk (jazz). All returned musically plausible neighbors.
