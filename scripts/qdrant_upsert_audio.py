#!/usr/bin/env python3
"""
qdrant_upsert_audio.py — SGG Audio Vector CLI
==============================================

Purpose
-------
This script is the primary tool for loading and querying audio feature vectors
in Qdrant. It reads from the dbt mart `fct_audio_vector_v1` (DuckDB) and manages
the `sgg_audio_v1` Qdrant collection, which stores 12-dimensional z-scored audio
feature vectors for every track in the collection.

It was validated in April 2026 against 4 seed tracks (Pink Floyd, Wolf Parade,
Norah Jones, Thelonious Monk) and confirmed that neighbors are musically plausible
across genre, energy, and mood dimensions.

Subcommands
-----------
- init   : create the Qdrant collection (size=12, cosine) and payload indexes
- stats  : compute per-feature means/stddevs from DuckDB and write to JSON
- upsert : load z-scored vectors from DuckDB into Qdrant with metadata payload
- query  : find top-k nearest neighbors for a given track (by file_hash)

Vector Features (12-dim, in order)
-----------------------------------
danceability, bpm, loudness_integrated, loudness_range, spectral_rms,
spectral_centroid, spectral_flux, pitch_salience, hfc, zcr, onset_rate,
key_strength

Usage Examples
--------------
  python scripts/qdrant_upsert_audio.py init
  python scripts/qdrant_upsert_audio.py stats
  python scripts/qdrant_upsert_audio.py upsert
  python scripts/qdrant_upsert_audio.py query --file-hash <hash> --topk 10

Testing Examples
----------------
  # Thelonious Monk — jazz contrast test
  python scripts/qdrant_upsert_audio.py query --file-hash 0071aaa53d57b3b5ea28dbd5fb68e86c --topk 10

Defaults
--------
- Source DB   : dbt_sgg/sgg_prod.duckdb
- Source view : fct_audio_vector_v1
- Collection  : sgg_audio_v1
- Vector      : 12-dim, cosine distance

Env (loaded from .env)
----------------------
- QDRANT_URL      : Qdrant REST endpoint (default: http://localhost:6335)
- QDRANT_API_KEY  : optional API key
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb  # type: ignore
from dotenv import load_dotenv  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.http import models as rest  # type: ignore


VECTOR_ORDER = [
    "danceability",
    "bpm",
    "loudness_integrated",
    "loudness_range",
    "spectral_rms",
    "spectral_centroid",
    "spectral_flux",
    "pitch_salience",
    "hfc",
    "zcr",
    "onset_rate",
    "key_strength",
]


def env_url() -> str:
    # Respect .env if present; fall back to 6335 (compose mapping) if not set
    url = os.environ.get("QDRANT_URL")
    if url:
        return url
    return "http://localhost:6335"


def load_stats(path: Optional[str]) -> Optional[Dict[str, Dict[str, float]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Stats file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_stats(path: str, stats: Dict[str, Dict[str, float]]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    return str(p)


def connect_duck(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def ensure_collection(client: QdrantClient, collection: str, size: int = 12, distance: str = "cosine") -> None:
    dist = rest.Distance.COSINE if distance.lower() == "cosine" else rest.Distance.DOT
    try:
        client.get_collection(collection)
        exists = True
    except Exception:
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(size=size, distance=dist),
        )
    # Add helpful payload indexes (ignore if already exist)
    for field in ("artist", "genre", "key_key", "key_scale", "vector_profile"):
        try:
            client.create_payload_index(collection, field, rest.PayloadSchemaType.KEYWORD)
        except Exception:
            pass


def compute_stats(con: duckdb.DuckDBPyConnection, view: str) -> Dict[str, Dict[str, float]]:
    cols = ", ".join(
        [
            f"avg({c}) as {c}_mean, stddev_samp({c}) as {c}_std" for c in VECTOR_ORDER
        ]
    )
    sql = f"select {cols} from {view}"
    row = con.execute(sql).fetchone()
    if row is None:
        raise SystemExit("No rows to compute stats")
    stats: Dict[str, Dict[str, float]] = {}
    # Map the result back by position
    values: List[float] = list(row)
    for i, feat in enumerate(VECTOR_ORDER):
        mean_v = values[2 * i]
        std_v = values[2 * i + 1]
        stats[feat] = {"mean": float(mean_v or 0.0), "std": float(std_v or 0.0)}
    return stats


def to_id(file_hash: str) -> int:
    # Use first 16 hex chars to fit in signed 64-bit range while keeping strong uniqueness
    try:
        return int(file_hash[:16], 16)
    except Exception:
        # Fallback: hash of full string
        return abs(hash(file_hash))


def build_vector_from_row(row: Dict[str, Any], *, vector_kind: str, stats: Optional[Dict[str, Dict[str, float]]]) -> Optional[List[float]]:
    if vector_kind == "raw":
        vec = []
        for f in VECTOR_ORDER:
            v = row.get(f)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return None
            vec.append(float(v))
        return vec
    # z vector
    if stats:
        vec: List[float] = []
        for f in VECTOR_ORDER:
            v = row.get(f)
            if v is None:
                return None
            mu = stats.get(f, {}).get("mean", 0.0)
            sd = stats.get(f, {}).get("std", 0.0) or 1.0
            vec.append((float(v) - float(mu)) / float(sd))
        return vec
    # fallback to precomputed vector_z column from dbt
    vcol = row.get("vector_z")
    if vcol is not None:
        try:
            as_list = list(vcol)
            if len(as_list) == len(VECTOR_ORDER):
                return [float(x) for x in as_list]
        except (TypeError, ValueError):
            pass
    return None


def iter_batches(con: duckdb.DuckDBPyConnection, view: str, limit: Optional[int], batch: int) -> Iterable[List[Dict[str, Any]]]:
    # Select required fields only
    fields = [
        "file_hash",
        "artist",
        "album",
        "title",
        "date",
        "genre",
        "key_key",
        "key_scale",
        "features_version",
        *VECTOR_ORDER,
        "vector_z",
    ]
    total = con.execute(f"select count(*) from {view}").fetchone()[0]
    to_read = min(total, limit) if limit else total
    offset = 0
    while offset < to_read:
        this = min(batch, to_read - offset)
        sql = f"select {', '.join(fields)} from {view} limit {this} offset {offset}"
        df = con.execute(sql).fetch_df()
        rows = df.to_dict(orient="records")
        yield rows
        offset += this


def action_init(args: argparse.Namespace) -> int:
    load_dotenv(override=True)
    client = QdrantClient(url=env_url(), api_key=os.environ.get("QDRANT_API_KEY") or None)
    ensure_collection(client, args.collection, size=12, distance=args.metric)
    print({"collection": args.collection, "metric": args.metric, "url": env_url()}, flush=True)
    return 0


def action_stats(args: argparse.Namespace) -> int:
    con = connect_duck(args.duckdb)
    stats = compute_stats(con, args.view)
    out = save_stats(args.out_json, stats)
    print({"wrote": out, "features": list(stats.keys())}, flush=True)
    return 0


def action_upsert(args: argparse.Namespace) -> int:
    load_dotenv(override=True)
    client = QdrantClient(url=env_url(), api_key=os.environ.get("QDRANT_API_KEY") or None)
    ensure_collection(client, args.collection, size=12, distance=args.metric)

    stats = load_stats(args.stats_json) if args.vector == "z" else None
    con = connect_duck(args.duckdb)
    total_sent = 0
    for rows in iter_batches(con, args.view, args.limit, args.batch_size):
        ids: List[int] = []
        vectors: List[List[float]] = []
        payloads: List[Dict[str, Any]] = []
        for r in rows:
            vec = build_vector_from_row(r, vector_kind=args.vector, stats=stats)
            if vec is None:
                continue
            fh = r.get("file_hash")
            if not fh:
                continue
            ids.append(to_id(str(fh)))
            vectors.append(vec)
            payloads.append(
                {
                    "file_hash": fh,
                    "artist": r.get("artist"),
                    "album": r.get("album"),
                    "title": r.get("title"),
                    "date": r.get("date"),
                    "genre": r.get("genre"),
                    "key_key": r.get("key_key"),
                    "key_scale": r.get("key_scale"),
                    "features_version": r.get("features_version"),
                    "vector_profile": "sgg_audio_v1",
                }
            )
        if not ids:
            continue
        batch = rest.Batch(ids=ids, vectors=vectors, payloads=payloads)
        client.upsert(collection_name=args.collection, points=batch)
        total_sent += len(ids)
        print({"upserted": len(ids), "total": total_sent}, flush=True)
    print({"done": True, "total": total_sent}, flush=True)
    return 0


def action_query(args: argparse.Namespace) -> int:
    load_dotenv(override=True)
    client = QdrantClient(url=env_url(), api_key=os.environ.get("QDRANT_API_KEY") or None)
    con = connect_duck(args.duckdb)
    # Get the source row
    row = (
        con.execute(
            f"select * from {args.view} where file_hash = ? limit 1", [args.file_hash]
        ).fetch_df().to_dict(orient="records")
    )
    if not row:
        print({"error": "file_hash not found in view", "file_hash": args.file_hash})
        return 1
    stats = load_stats(args.stats_json) if args.vector == "z" else None
    vec = build_vector_from_row(row[0], vector_kind=args.vector, stats=stats)
    if not vec:
        print({"error": "could not build vector"})
        return 1
    # Optional filter on payload
    flt: Optional[rest.Filter] = None
    if args.genre or args.key_key or args.key_scale:
        must: List[rest.FieldCondition] = []
        if args.genre:
            must.append(rest.FieldCondition(key="genre", match=rest.MatchValue(value=args.genre)))
        if args.key_key:
            must.append(rest.FieldCondition(key="key_key", match=rest.MatchValue(value=args.key_key)))
        if args.key_scale:
            must.append(rest.FieldCondition(key="key_scale", match=rest.MatchValue(value=args.key_scale)))
        if must:
            flt = rest.Filter(must=must)

    res = client.search(
        collection_name=args.collection,
        query_vector=vec,
        limit=args.topk,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )
    out = [
        {
            "id": int(p.id),
            "score": float(p.score or 0.0),
            "file_hash": p.payload.get("file_hash"),
            "artist": p.payload.get("artist"),
            "title": p.payload.get("title"),
            "album": p.payload.get("album"),
            "genre": p.payload.get("genre"),
            "key": f"{p.payload.get('key_key')}-{p.payload.get('key_scale')}",
        }
        for p in res
    ]
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SGG → Qdrant upsert CLI")
    ap.add_argument("--duckdb", default="dbt_sgg/sgg_prod.duckdb", help="DuckDB file path")
    ap.add_argument("--view", default="fct_audio_vector_v1", help="Source view name")
    ap.add_argument("--collection", default="sgg_audio_v1", help="Qdrant collection name")
    ap.add_argument("--metric", default="cosine", choices=["cosine", "dot"], help="Vector distance metric")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Ensure collection and payload indexes exist")
    p_init.set_defaults(func=action_init)

    p_stats = sub.add_parser("stats", help="Compute and write vector stats JSON")
    p_stats.add_argument("--out-json", default="data/features/sgg_audio_v1_stats.json")
    p_stats.set_defaults(func=action_stats)

    p_up = sub.add_parser("upsert", help="Upsert vectors into Qdrant from DuckDB")
    p_up.add_argument("--vector", default="z", choices=["z", "raw"], help="Which vector to use")
    p_up.add_argument("--stats-json", default=None, help="Stats JSON for z-scores; if omitted, uses dbt vector_z")
    p_up.add_argument("--batch-size", type=int, default=1000)
    p_up.add_argument("--limit", type=int, default=None)
    p_up.set_defaults(func=action_upsert)

    p_q = sub.add_parser("query", help="Search top-k by file_hash; optional payload filters")
    p_q.add_argument("--file-hash", required=True)
    p_q.add_argument("--vector", default="z", choices=["z", "raw"]) 
    p_q.add_argument("--stats-json", default=None)
    p_q.add_argument("--topk", type=int, default=10)
    p_q.add_argument("--genre", default=None)
    p_q.add_argument("--key_key", default=None)
    p_q.add_argument("--key_scale", default=None)
    p_q.set_defaults(func=action_query)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

