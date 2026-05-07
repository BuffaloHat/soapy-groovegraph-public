#!/usr/bin/env python3
"""
eval_audio_similarity.py — SGG Audio Similarity Evaluation Export
==================================================================

Purpose
-------
Runs top-k audio similarity queries against the sgg_audio_v1 Qdrant collection
for a set of hand-picked seed tracks and exports results to a CSV for spot-check
review. This documents that the audio similarity engine was tested and produces
musically plausible neighbors.

Seeds used for April 2026 validation:
  - Pink Floyd    : "Speak To Me/Breathe"   (atmospheric rock)
  - Wolf Parade   : "We Built Another World" (high-energy indie)
  - Norah Jones   : "Be Here To Love Me"     (quiet acoustic pop)
  - Thelonious Monk: "San Francisco Holiday" (jazz)

Output
------
  data/eval/audio_similarity_eval_2026.csv

Usage
-----
  python scripts/eval_audio_similarity.py
  python scripts/eval_audio_similarity.py --topk 10 --out data/eval/my_eval.csv

Env (loaded from .env)
----------------------
  QDRANT_URL     : Qdrant REST endpoint (default: http://localhost:6335)
  QDRANT_API_KEY : optional
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

VECTOR_ORDER = [
    "danceability", "bpm", "loudness_integrated", "loudness_range",
    "spectral_rms", "spectral_centroid", "spectral_flux", "pitch_salience",
    "hfc", "zcr", "onset_rate", "key_strength",
]

SEEDS = [
    {"file_hash": "02a0cfab86c3adef2d5e3e9a14f60e43", "label": "Pink Floyd — atmospheric rock"},
    {"file_hash": "005ace98693e050435f6b4c46820815b", "label": "Wolf Parade — high-energy indie"},
    {"file_hash": "02594d241327e2e5dd1386239552b507", "label": "Norah Jones — quiet acoustic pop"},
    {"file_hash": "0071aaa53d57b3b5ea28dbd5fb68e86c", "label": "Thelonious Monk — jazz"},
]

DUCKDB_PATH = "dbt_sgg/sgg_prod.duckdb"
VIEW = "fct_audio_vector_v1"
COLLECTION = "sgg_audio_v1"


def get_vector(con: duckdb.DuckDBPyConnection, file_hash: str) -> list[float] | None:
    df = con.execute(
        f"select * from {VIEW} where file_hash = ? limit 1", [file_hash]
    ).fetch_df()
    if df.empty:
        return None
    row = df.iloc[0]
    vcol = row.get("vector_z")
    if vcol is not None:
        try:
            as_list = list(vcol)
            if len(as_list) == len(VECTOR_ORDER):
                return [float(x) for x in as_list]
        except (TypeError, ValueError):
            pass
    return None


def get_seed_meta(con: duckdb.DuckDBPyConnection, file_hash: str) -> dict:
    df = con.execute(
        f"select artist, title, album, genre, key_key, key_scale from {VIEW} where file_hash = ? limit 1",
        [file_hash]
    ).fetch_df()
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def run_eval(topk: int, out_path: str) -> None:
    load_dotenv(override=True)

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6335")
    client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in SEEDS:
        fh = seed["file_hash"]
        label = seed["label"]
        print(f"Querying: {label} ({fh[:8]}...)")

        vec = get_vector(con, fh)
        if vec is None:
            print(f"  WARNING: could not build vector for {fh}, skipping.")
            continue

        meta = get_seed_meta(con, fh)

        results = client.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=topk,
            with_payload=True,
            with_vectors=False,
        ).points

        for rank, point in enumerate(results, start=1):
            p = point.payload or {}
            rows.append({
                "seed_label": label,
                "seed_file_hash": fh,
                "seed_artist": meta.get("artist", ""),
                "seed_title": meta.get("title", ""),
                "seed_album": meta.get("album", ""),
                "seed_genre": meta.get("genre", ""),
                "seed_key": f"{meta.get('key_key', '')}-{meta.get('key_scale', '')}",
                "rank": rank,
                "score": round(float(point.score or 0.0), 6),
                "neighbor_file_hash": p.get("file_hash", ""),
                "neighbor_artist": p.get("artist", ""),
                "neighbor_title": p.get("title", ""),
                "neighbor_album": p.get("album", ""),
                "neighbor_genre": p.get("genre", ""),
                "neighbor_key": f"{p.get('key_key', '')}-{p.get('key_scale', '')}",
                "is_self": fh == p.get("file_hash", ""),
            })

    fieldnames = [
        "seed_label", "seed_file_hash", "seed_artist", "seed_title", "seed_album",
        "seed_genre", "seed_key", "rank", "score", "neighbor_file_hash",
        "neighbor_artist", "neighbor_title", "neighbor_album", "neighbor_genre",
        "neighbor_key", "is_self",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExported {len(rows)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export audio similarity eval CSV")
    parser.add_argument("--topk", type=int, default=10, help="Neighbors per seed (default: 10)")
    parser.add_argument("--out", default="data/eval/audio_similarity_eval_2026.csv", help="Output CSV path")
    args = parser.parse_args()
    run_eval(topk=args.topk, out_path=args.out)


if __name__ == "__main__":
    main()
