#!/usr/bin/env python3
"""
sgg_text_embed.py — SGG Text Embedding CLI
==========================================

Purpose
-------
Generates text embeddings for every track in the collection and upserts them
into the `sgg_text_v1` Qdrant collection. This powers the Text RAG feature —
allowing natural-language questions to retrieve relevant tracks by meaning
rather than exact keyword match.

Each track is converted into a natural-language sentence combining its metadata
and key audio characteristics, then embedded via Ollama (nomic-embed-text).

Example document per track:
  "Ripple by Grateful Dead, from the album Reckoning (1981). Genre: Other.
   Key: G major, brighter and more uplifting. Tempo: slow at 76 BPM.
   Energy: quiet and gentle. Groove: moderate groove. Dynamics: wide dynamic range."

Change Log
----------
2026-04-29 — Richer build_doc function (re-embedded all 9,254 tracks)
  Problem: The original track descriptions were too generic. Every track got one of only
  three danceability labels and three energy labels, meaning thousands of tracks ended up
  with nearly identical descriptions. This caused tracks to cluster too tightly in the
  embedding space, making some of them "hubs" that appeared in results for almost any query.

  Fix: The build_doc function was redesigned to produce more descriptive, discriminative text:

  - Key mood: Instead of just "A minor", the doc now says "A minor, darker and more
    melancholic" (or "major, brighter and more uplifting"). This helps the embedding model
    match mood-based queries like "melancholy music" or "uplifting songs" correctly.

  - Tempo labels: BPM is now expressed as a word ("slow", "moderate tempo", "upbeat",
    "fast", etc.) alongside the raw number, using thresholds calibrated to standard musical
    tempo terminology. The old doc just printed the raw BPM with no context.

  - More granular energy: Loudness is now split into 5 buckets instead of 3, so "quiet and
    gentle" and "very quiet and intimate" are distinct — previously both mapped to the same
    label.

  - More granular groove: Danceability thresholds were recalibrated against the actual
    library distribution (which clusters tightly between 1.0 and 1.4) to create 5 meaningful
    buckets instead of 3 coarse ones.

  - Dynamics: A new field (loudness_range from Essentia) was added to describe whether a
    track has a "compressed, consistent sound" or "wide dynamic range". This was not in the
    original doc at all.

  After rebuilding all docs, the embed subcommand was re-run to regenerate and upsert all
  vectors into sgg_text_v1.

Subcommands
-----------
- init   : create the sgg_text_v1 Qdrant collection
- embed  : generate embeddings and upsert to Qdrant
- query  : search top-k neighbors for a plain-text query string

Usage
-----
  python scripts/sgg_text_embed.py init
  python scripts/sgg_text_embed.py embed
  python scripts/sgg_text_embed.py embed --limit 100   # test on subset first
  python scripts/sgg_text_embed.py query --text "slow jazzy late night music"

Env (loaded from .env)
----------------------
  QDRANT_URL             : Qdrant REST endpoint (default: http://localhost:6335)
  QDRANT_API_KEY         : optional
  OLLAMA_BASE_URL_HOST   : Ollama endpoint (default: http://localhost:11434)
  OLLAMA_EMBED_MODEL     : embedding model (default: nomic-embed-text)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import duckdb
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

DUCKDB_PATH = "dbt_sgg/sgg_prod.duckdb"
VIEW = "fct_audio_vector_v1"
COLLECTION = "sgg_text_v1"
EMBED_DIM = 768  # nomic-embed-text output dimension


def qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6335")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)


def ollama_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL_HOST", "http://localhost:11434")


def embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{ollama_url()}/api/embeddings",
        json={"model": embed_model(), "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def tempo_label(bpm: float) -> str:
    if bpm < 65: return "very slow"
    elif bpm < 85: return "slow"
    elif bpm < 105: return "moderate tempo"
    elif bpm < 125: return "upbeat"
    elif bpm < 145: return "fast"
    else: return "very fast"


def energy_label(loudness: float) -> str:
    if loudness < -20: return "very quiet and intimate"
    elif loudness < -17: return "quiet and gentle"
    elif loudness < -14: return "moderate energy"
    elif loudness < -11: return "energetic"
    else: return "loud and driving"


def danceability_label(val: float) -> str:
    # Library clusters tightly: p10=0.98, p50=1.16, p90=1.38; values above ~1.5 are outliers
    if val < 1.0: return "low groove"
    elif val < 1.1: return "moderate groove"
    elif val < 1.2: return "groovy"
    elif val < 1.35: return "danceable"
    else: return "highly danceable"


def dynamics_label(loudness_range: float) -> str:
    # p10=2.76, p50=5.56, p90=11.46
    if loudness_range < 4.0: return "compressed, consistent sound"
    elif loudness_range < 8.0: return "moderate dynamics"
    else: return "wide dynamic range"


def key_description(key: str, scale: str) -> str:
    if not key:
        return ""
    if scale and scale.lower() == "minor":
        return f"{key} minor, darker and more melancholic"
    elif scale and scale.lower() == "major":
        return f"{key} major, brighter and more uplifting"
    return f"{key} {scale}".strip()


def build_doc(row: dict) -> str:
    title = row.get("title") or "Unknown Title"
    artist = row.get("artist") or "Unknown Artist"
    album = row.get("album") or "Unknown Album"
    year = str(row.get("date") or "").split("-")[0] or "unknown year"
    genre = row.get("genre") or "Unknown Genre"
    bpm = row.get("bpm")
    danceability = row.get("danceability")
    loudness = row.get("loudness_integrated")
    loudness_range = row.get("loudness_range")

    key_str = key_description(row.get("key_key") or "", row.get("key_scale") or "")
    tempo_str = f"Tempo: {tempo_label(bpm)} at {int(round(bpm))} BPM." if bpm is not None else ""
    energy_str = f"Energy: {energy_label(loudness)}." if loudness is not None else ""
    groove_str = f"Groove: {danceability_label(danceability)}." if danceability is not None else ""
    dynamics_str = f"Dynamics: {dynamics_label(loudness_range)}." if loudness_range is not None else ""

    return (
        f"{title} by {artist}, from the album {album} ({year}). "
        f"Genre: {genre}. "
        f"Key: {key_str}. "
        f"{tempo_str} {energy_str} {groove_str} {dynamics_str}"
    ).strip()


def to_point_id(file_hash: str) -> int:
    try:
        return int(file_hash[:16], 16)
    except Exception:
        return abs(hash(file_hash))


def action_init(args: argparse.Namespace) -> None:
    client = qdrant_client()
    try:
        client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' already exists.")
        return
    except Exception:
        pass
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=rest.VectorParams(size=EMBED_DIM, distance=rest.Distance.COSINE),
    )
    for field in ("artist", "genre", "key_key", "key_scale"):
        try:
            client.create_payload_index(COLLECTION, field, rest.PayloadSchemaType.KEYWORD)
        except Exception:
            pass
    print(f"Collection '{COLLECTION}' created (dim={EMBED_DIM}, cosine).")


def action_embed(args: argparse.Namespace) -> None:
    client = qdrant_client()
    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    fields = "file_hash, artist, album, title, date, genre, key_key, key_scale, bpm, danceability, loudness_integrated, loudness_range"
    limit_clause = f"limit {args.limit}" if args.limit else ""
    rows = con.execute(f"select {fields} from {VIEW} {limit_clause}").fetch_df().to_dict(orient="records")

    total = len(rows)
    print(f"Embedding {total} tracks via {embed_model()} ...")

    points = []
    for i, row in enumerate(rows, start=1):
        fh = row.get("file_hash")
        if not fh:
            continue
        doc = build_doc(row)
        try:
            vector = get_embedding(doc)
        except Exception as e:
            print(f"  [{i}/{total}] ERROR embedding {fh[:8]}: {e}")
            continue

        points.append(rest.PointStruct(
            id=to_point_id(str(fh)),
            vector=vector,
            payload={
                "file_hash": fh,
                "artist": row.get("artist"),
                "album": row.get("album"),
                "title": row.get("title"),
                "date": str(row.get("date") or ""),
                "genre": row.get("genre"),
                "key_key": row.get("key_key"),
                "key_scale": row.get("key_scale"),
                "doc": doc,
            },
        ))

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] upserting batch ...")
            client.upsert(collection_name=COLLECTION, points=points)
            points = []

    print(f"Done. {total} tracks embedded and upserted to '{COLLECTION}'.")


def action_query(args: argparse.Namespace) -> None:
    client = qdrant_client()
    print(f"Embedding query: '{args.text}'")
    vector = get_embedding(args.text)

    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=args.topk,
        with_payload=True,
        with_vectors=False,
    ).points

    print(f"\nTop {args.topk} results:\n")
    for i, point in enumerate(results, start=1):
        p = point.payload or {}
        print(f"  {i}. [{round(point.score, 4)}] {p.get('artist')} — {p.get('title')} ({p.get('album')})")
        print(f"     Genre: {p.get('genre')}  Key: {p.get('key_key')}-{p.get('key_scale')}")
        print(f"     Doc: {p.get('doc')}")
        print()


def chat_model() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "gemma3:27b")


def action_rag(args: argparse.Namespace) -> None:
    client = qdrant_client()

    print(f"Query: \"{args.text}\"\n")
    vector = get_embedding(args.text)

    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=args.topk,
        with_payload=True,
        with_vectors=False,
    ).points

    context_lines = []
    for i, point in enumerate(results, start=1):
        p = point.payload or {}
        context_lines.append(
            f"{i}. {p.get('artist')} — {p.get('title')} ({p.get('album')})\n"
            f"   {p.get('doc')}"
        )
    context = "\n".join(context_lines)

    print("--- Retrieved Tracks ---")
    print(context)
    print()

    prompt = (
        f"You are a music recommendation assistant. A user asked: \"{args.text}\"\n\n"
        f"Here are the most relevant tracks from their vinyl collection:\n\n"
        f"{context}\n\n"
        f"Based only on the tracks above, write a short, friendly recommendation. "
        f"Cite specific tracks by artist and title. Do not invent tracks that aren't listed."
    )

    print(f"--- Generating answer with {chat_model()} ---\n")
    resp = requests.post(
        f"{ollama_url()}/api/chat",
        json={
            "model": chat_model(),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    print(resp.json()["message"]["content"])


def main() -> None:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="SGG text embedding CLI for sgg_text_v1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Create sgg_text_v1 collection in Qdrant").set_defaults(func=action_init)

    p_embed = sub.add_parser("embed", help="Embed all tracks and upsert to Qdrant")
    p_embed.add_argument("--limit", type=int, default=None, help="Limit rows (for testing)")
    p_embed.set_defaults(func=action_embed)

    p_query = sub.add_parser("query", help="Search by plain-text query")
    p_query.add_argument("--text", required=True, help="Natural language query")
    p_query.add_argument("--topk", type=int, default=5)
    p_query.set_defaults(func=action_query)

    p_rag = sub.add_parser("rag", help="Retrieve tracks and generate a cited answer via LLM")
    p_rag.add_argument("--text", required=True, help="Natural language question")
    p_rag.add_argument("--topk", type=int, default=8)
    p_rag.set_defaults(func=action_rag)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
