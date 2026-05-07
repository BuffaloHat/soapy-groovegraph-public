#!/usr/bin/env python3
"""
Flatten Essentia extractor JSON files to a Parquet dataset suitable for dbt/DuckDB and Qdrant prep.

Inputs
- Per-track JSON files from `es_run_extractor.py` under `data/raw/essentia/extraction/*.json`

Outputs
- Parquet dataset directory: `data/raw/essentia/essentia_features_v1/part-*.parquet`
- (Optional) sample Parquet when `--sample-parquet` is passed.

Notes
- Schema is v1 with nullable placeholders for mood and voice/instrumental classifiers (require Essentia high-level models, not run in v1).
- Tag fields like artist/album/title/genre/etc are normalized to scalars for BI/dbt friendliness.
- Numeric vectors `mfcc_mean` and `thpcp` stay as arrays.
- Requires `pyarrow` for Parquet IO. Install into the repo venv: `pip install pyarrow`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        s = str(x)
        # handle forms like "03/12"
        if "/" in s:
            s = s.split("/", 1)[0]
        s = s.strip()
        return int(s)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _first_string(value: Any) -> Optional[str]:
    """Return a scalar string from value that may already be a string or a list of strings.
    Picks the first non-empty string; returns None if nothing usable.
    """
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v if v != "" else None
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    return s
    # Fallback to string-ified scalar
    try:
        s = str(value).strip()
        return s if s else None
    except Exception:
        return None


def list_to_csv(values: Optional[Iterable[str]]) -> Optional[str]:
    if values is None:
        return None
    out: List[str] = []
    seen = set()
    for v in values:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return ", ".join(out) if out else None


def _first_existing(d: Dict[str, Any], paths: List[str]) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Return (key, scale, strength) from the first available tonal key block.
    Prefers `tonal.key_edma`, then `tonal.key_krumhansl`, then `tonal.key_temperley`.
    """
    for p in paths:
        blk = _get(d, p)
        if isinstance(blk, dict):
            key = blk.get("key")
            scale = blk.get("scale")
            strength = blk.get("strength")
            if key or scale or strength is not None:
                return key, scale, strength
    return None, None, None


def flatten_essentia_json(obj: Dict[str, Any], json_path: str) -> Dict[str, Any]:
    tags = _get(obj, "metadata.tags", {}) or {}
    ap = _get(obj, "metadata.audio_properties", {}) or {}
    ver = _get(obj, "metadata.version", {}) or {}

    # Identity / metadata
    artist = _first_string(tags.get("artist"))
    album = _first_string(tags.get("album"))
    title = _first_string(tags.get("title"))
    date = _first_string(tags.get("date"))
    # Genre can be multi-valued in some dumps; keep scalar + csv
    genre_val = tags.get("genre")
    if isinstance(genre_val, list):
        genre = _first_string(genre_val)
        genre_csv = list_to_csv(genre_val)
    else:
        genre = _first_string(genre_val)
        genre_csv = genre
    albumartist = _first_string(tags.get("albumartist"))
    composer = _first_string(tags.get("composer"))
    tracknumber = _to_int(tags.get("tracknumber"))
    discnumber = _to_int(tags.get("discnumber"))
    file_name = _first_string(tags.get("file_name"))

    # Rhythm
    bpm = _get(obj, "rhythm.bpm")
    danceability = _get(obj, "rhythm.danceability")
    onset_rate = _get(obj, "rhythm.onset_rate")
    beats_count = _get(obj, "rhythm.beats_count")
    bpm_peak1 = _get(obj, "rhythm.bpm_histogram_first_peak_bpm")
    bpm_peak2 = _get(obj, "rhythm.bpm_histogram_second_peak_bpm")

    # Tonal / key
    key_key, key_scale, key_strength = _first_existing(
        obj,
        [
            "tonal.key_edma",
            "tonal.key_krumhansl",
            "tonal.key_temperley",
        ],
    )
    tuning_frequency = _get(obj, "tonal.tuning_frequency")
    equal_tempered_deviation = _get(obj, "tonal.tuning_equal_tempered_deviation")
    chords_changes_rate = _get(obj, "tonal.chords_changes_rate")

    # Loudness / dynamics
    loudness_integrated = _get(obj, "lowlevel.loudness_ebu128.integrated")
    loudness_range = _get(obj, "lowlevel.loudness_ebu128.loudness_range")
    avg_loudness = _get(obj, "lowlevel.average_loudness")
    dyn_complexity = _get(obj, "lowlevel.dynamic_complexity")

    # Spectral / timbral summaries
    spectral_rms = _get(obj, "lowlevel.spectral_rms.mean")
    spectral_centroid = _get(obj, "lowlevel.spectral_centroid.mean")
    spectral_flux = _get(obj, "lowlevel.spectral_flux.mean")
    spectral_spread = _get(obj, "lowlevel.spectral_spread.mean")
    pitch_salience = _get(obj, "lowlevel.pitch_salience.mean")
    hfc = _get(obj, "lowlevel.hfc.mean")
    zcr = _get(obj, "lowlevel.zerocrossingrate.mean")

    # Vectors
    mfcc_mean = _get(obj, "lowlevel.mfcc.mean")
    thpcp = _get(obj, "tonal.thpcp")

    # Audio properties / provenance
    duration_sec = ap.get("length")
    sample_rate = ap.get("sample_rate")
    bit_rate = ap.get("bit_rate")
    channels = ap.get("number_channels")
    codec = ap.get("codec")
    lossless = ap.get("lossless")
    replay_gain = ap.get("replay_gain")
    file_hash = ap.get("md5_encoded")

    essentia_version = ver.get("essentia")
    extracted_with = ver.get("extractor")
    features_version = "essentia_music_v1"

    # Highlevel placeholders (not present in current dumps)
    voice_instrumental_value = None
    voice_instrumental_prob_vocal = None
    voice_instrumental_prob_instrumental = None
    mood_happy = None
    mood_relaxed = None
    mood_party = None
    mood_model = None
    mood_version = None

    extracted_at = datetime.utcfromtimestamp(os.path.getmtime(json_path)).isoformat() + "Z"

    row = {
        # Identity/metadata
        "artist": artist,
        "album": album,
        "title": title,
        "date": date,
        "genre": genre,
        "genre_csv": genre_csv,
        "albumartist": albumartist,
        "composer": composer,
        "tracknumber": tracknumber,
        "discnumber": discnumber,
        "file_name": file_name,
        # Audio properties
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "bit_rate": bit_rate,
        "channels": channels,
        "codec": codec,
        "lossless": lossless,
        "replay_gain": replay_gain,
        # Rhythm
        "bpm": bpm,
        "danceability": danceability,
        "onset_rate": onset_rate,
        "beats_count": beats_count,
        "bpm_peak1": bpm_peak1,
        "bpm_peak2": bpm_peak2,
        # Key/tonal
        "key_key": key_key,
        "key_scale": key_scale,
        "key_strength": key_strength,
        "tuning_frequency": tuning_frequency,
        "equal_tempered_deviation": equal_tempered_deviation,
        "chords_changes_rate": chords_changes_rate,
        # Loudness/dynamics
        "loudness_integrated": loudness_integrated,
        "loudness_range": loudness_range,
        "avg_loudness": avg_loudness,
        "dyn_complexity": dyn_complexity,
        # Spectral/timbral
        "spectral_rms": spectral_rms,
        "spectral_centroid": spectral_centroid,
        "spectral_flux": spectral_flux,
        "spectral_spread": spectral_spread,
        "pitch_salience": pitch_salience,
        "hfc": hfc,
        "zcr": zcr,
        # Vectors (arrays)
        "mfcc_mean": mfcc_mean,
        "thpcp": thpcp,
        # Highlevel placeholders
        "voice_instrumental_value": voice_instrumental_value,
        "voice_instrumental_prob_vocal": voice_instrumental_prob_vocal,
        "voice_instrumental_prob_instrumental": voice_instrumental_prob_instrumental,
        "mood_happy": mood_happy,
        "mood_relaxed": mood_relaxed,
        "mood_party": mood_party,
        "mood_model": mood_model,
        "mood_version": mood_version,
        # Provenance / ids
        "features_version": features_version,
        "essentia_version": essentia_version,
        "extracted_with": extracted_with,
        "source_path": None,  # not available in current JSON; kept for future
        "file_hash": file_hash,
        "json_path": json_path,
        "extracted_at": extracted_at,
    }
    return row


def scan_json_files(in_dir: str, limit: Optional[int] = None) -> List[str]:
    paths = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    if limit is not None:
        return paths[: int(limit)]
    return paths


def write_dataset_parts(df: pd.DataFrame, out_dir: str, chunk_rows: int = 5000) -> List[str]:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("pyarrow is required. Install with: pip install pyarrow")

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    n = len(df)
    if n == 0:
        return paths
    part_idx = 0
    for start in range(0, n, max(1, chunk_rows)):
        end = min(n, start + chunk_rows)
        part = df.iloc[start:end]
        part_path = os.path.join(out_dir, f"part-{part_idx:05d}.parquet")
        part.to_parquet(part_path, index=False)
        paths.append(part_path)
        part_idx += 1
    return paths


def write_sample(df: pd.DataFrame, sample_path: str, sample_rows: int = 500) -> str:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("pyarrow is required. Install with: pip install pyarrow")
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    df.head(sample_rows).to_parquet(sample_path, index=False)
    return sample_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Flatten Essentia JSON to Parquet dataset")
    ap.add_argument("--in-dir", default="data/raw/essentia/extraction", help="Folder with Essentia JSON files")
    ap.add_argument(
        "--out-dataset",
        default="data/raw/essentia/essentia_features_v1",
        help="Output folder for Parquet dataset (part-*.parquet)",
    )
    ap.add_argument(
        "--out-parquet",
        default=None,
        help="Optional single Parquet file path; if set, writes a single file instead of a dataset",
    )
    ap.add_argument(
        "--sample-parquet",
        default=None,
        help="Optional small Parquet sample for inspection (omit to skip)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Process only first N files (debug)")
    ap.add_argument("--chunk-rows", type=int, default=5000, help="Rows per part file")
    ap.add_argument("--sample-rows", type=int, default=500, help="Rows in sample parquet")
    ap.add_argument("--overwrite", action="store_true", help="Clear output dir before writing")
    args = ap.parse_args(argv)

    json_paths = scan_json_files(args.in_dir, args.limit)
    if not json_paths:
        print(f"No JSON files found under {args.in_dir}")
        return 1

    rows: List[Dict[str, Any]] = []
    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            rows.append(flatten_essentia_json(obj, jp))
        except Exception as e:
            print(f"WARN: failed {jp}: {e}")
            continue

    df = pd.DataFrame(rows)

    if args.overwrite and os.path.isdir(args.out_dataset):
        # Remove existing part files (non-destructive to parent dirs)
        for old in glob.glob(os.path.join(args.out_dataset, "part-*.parquet")):
            try:
                os.remove(old)
            except Exception:
                pass

    part_paths: List[str] = []
    if args.out_parquet:
        # Write monolithic Parquet file
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise SystemExit("pyarrow is required. Install with: pip install pyarrow")
        os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
        df.to_parquet(args.out_parquet, index=False)
    else:
        part_paths = write_dataset_parts(df, args.out_dataset, args.chunk_rows)
    sample_path: Optional[str] = None
    if args.sample_parquet:
        sample_path = write_sample(df, args.sample_parquet, args.sample_rows)

    if args.out_parquet:
        print(
            "Created parquet",
            {
                "rows": len(df),
                "out_file": args.out_parquet,
                **({"sample": sample_path} if sample_path else {}),
            },
            flush=True,
        )
    else:
        print(
            "Created dataset",
            {
                "parts": len(part_paths),
                "rows": len(df),
                "out_dir": args.out_dataset,
                **({"sample": sample_path} if sample_path else {}),
            },
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
