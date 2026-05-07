"""
iTunes image lookup pipeline.

Resolves album artwork URLs for all unique artist+album pairs in the Essentia
dataset, caches results to data/cache/itunes_image_cache.json, and flattens
to Parquet. Skips pairs already in the cache so reruns are safe.

Cache values:
  "<url>"  — image resolved successfully
  null     — genuine miss (no album art found on iTunes)
  "RETRY"  — API error (rate limit, timeout) — will be retried on next run

Usage:
    python scripts/itunes_image_lookup.py            # run / resume full lookup
    python scripts/itunes_image_lookup.py --stats    # print cache hit rate only
    python scripts/itunes_image_lookup.py --limit 50 # test run on first N pairs
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path("data/cache/itunes_image_cache.json")
PARQUET_PATH = Path("data/cache/itunes_image_cache.parquet")
DB_PATH = Path("dbt_sgg/sgg_prod.duckdb")
REQUEST_DELAY = 2.0       # seconds between requests
CHUNK_SIZE = 50           # requests per chunk before a long pause
CHUNK_PAUSE = 90          # seconds to pause between chunks
RETRY_SENTINEL = "RETRY"  # stored when a request fails — retried on next run


def _cache_key(artist: str, album: str) -> str:
    return f"{artist.lower().strip()}|{album.lower().strip()}"


def _clean_album(album: str) -> str:
    album = re.sub(r'\s*[\(\[]disc\s*\d+[\)\]]', '', album, flags=re.IGNORECASE)
    album = re.sub(r'\s*[\(\[](live|.*?live.*?)[\)\]]$', '', album, flags=re.IGNORECASE)
    album = re.sub(r'\s*-\s*(EP|Single|Remaster(ed)?|Deluxe.*|Expanded.*|Anniversary.*)$', '', album, flags=re.IGNORECASE)
    album = re.sub(r'\s*[\(\[](Remaster(ed)?|Deluxe.*|Expanded.*|Anniversary.*)[\)\]]', '', album, flags=re.IGNORECASE)
    return album.strip()


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def save_parquet(cache: dict) -> None:
    rows = []
    for key, url in cache.items():
        if url == RETRY_SENTINEL:
            continue
        artist, album = key.split("|", 1)
        rows.append({"cache_key": key, "artist": artist, "album": album, "artwork_url": url})
    df = pd.DataFrame(rows)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"  Parquet written: {PARQUET_PATH} ({len(df)} rows)")


def itunes_search(artist: str, album: str) -> Optional[str]:
    """Returns URL string, None (genuine miss), or raises urllib.error.HTTPError on 429."""
    clean = _clean_album(album)
    term = urllib.parse.quote(f"{artist} {clean}")
    url = f"https://itunes.apple.com/search?term={term}&entity=album&limit=5&media=music"
    with urllib.request.urlopen(url, timeout=8) as r:
        data = json.loads(r.read())
    for result in data.get("results", []):
        if result.get("wrapperType") == "collection":
            art = result.get("artworkUrl100", "")
            if art:
                return art.replace("100x100bb", "600x600bb")
    return None


def load_pairs() -> list:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT DISTINCT artist, album
        FROM im_essentia_features_unique
        WHERE artist IS NOT NULL AND artist != ''
          AND album  IS NOT NULL AND album  != ''
        ORDER BY artist, album
    """).fetchdf()
    con.close()
    return list(df.itertuples(index=False, name=None))


def print_stats(cache: dict, total_pairs: int) -> None:
    hits    = sum(1 for v in cache.values() if v and v != RETRY_SENTINEL)
    misses  = sum(1 for v in cache.values() if v is None)
    retries = sum(1 for v in cache.values() if v == RETRY_SENTINEL)
    cached  = len(cache)
    resolved = hits + misses
    print(f"\n=== iTunes Image Cache Stats ===")
    print(f"  Total unique artist+album pairs : {total_pairs}")
    print(f"  Resolved (hits + misses)        : {resolved}")
    print(f"  Hits  (image found)             : {hits}  ({hits/resolved*100:.1f}% of resolved)" if resolved else "")
    print(f"  Misses (no image on iTunes)     : {misses}")
    print(f"  Pending retry (API errors)      : {retries}")
    print(f"  Not yet looked up               : {total_pairs - cached}")
    print(f"  Cache path                      : {CACHE_PATH}")


def run(limit: Optional[int] = None) -> None:
    print("Loading Essentia artist/album pairs...")
    pairs = load_pairs()
    if limit:
        pairs = pairs[:limit]
        print(f"  Limiting to first {limit} pairs (test mode)")

    cache = load_cache()
    pending = [(a, b) for a, b in pairs if cache.get(_cache_key(a, b)) in (None.__class__.__name__, RETRY_SENTINEL) or _cache_key(a, b) not in cache]
    # rebuild: skip only confirmed hits and genuine misses (None), retry RETRY_SENTINEL entries
    pending = [(a, b) for a, b in pairs if _cache_key(a, b) not in cache or cache[_cache_key(a, b)] == RETRY_SENTINEL]

    print(f"  {len(pairs)} pairs total — {len(cache) - sum(1 for v in cache.values() if v == RETRY_SENTINEL)} resolved — {len(pending)} to process\n")

    new_hits = 0
    new_misses = 0
    consecutive_errors = 0

    for i, (artist, album) in enumerate(pending, 1):
        key = _cache_key(artist, album)

        try:
            url = itunes_search(artist, album)
            cache[key] = url
            consecutive_errors = 0

            if url:
                new_hits += 1
                status = "HIT"
            else:
                new_misses += 1
                status = "MISS"

        except urllib.error.HTTPError as e:
            if e.code == 429:
                consecutive_errors += 1
                backoff = min(30, 5 * consecutive_errors)
                print(f"    RATE LIMITED — backing off {backoff}s (consecutive: {consecutive_errors})")
                cache[key] = RETRY_SENTINEL
                time.sleep(backoff)
                status = "RETRY"
            else:
                print(f"    WARNING: HTTP {e.code} for '{artist} / {album}'")
                cache[key] = RETRY_SENTINEL
                status = "RETRY"

        except Exception as e:
            print(f"    WARNING: error for '{artist} / {album}': {e}")
            cache[key] = RETRY_SENTINEL
            status = "RETRY"

        print(f"  [{i:>4}/{len(pending)}] [{status}] {artist} — {album}")

        if i % CHUNK_SIZE == 0:
            save_cache(cache)
            print(f"  --- checkpoint saved ({i} processed) — pausing {CHUNK_PAUSE}s to respect rate limit ---")
            time.sleep(CHUNK_PAUSE)
        else:
            time.sleep(REQUEST_DELAY)

    save_cache(cache)
    save_parquet(cache)

    retries_remaining = sum(1 for v in cache.values() if v == RETRY_SENTINEL)
    print(f"\n  Done. New hits: {new_hits} | New misses: {new_misses} | Retries remaining: {retries_remaining}")
    print_stats(cache, len(load_pairs()))
    if retries_remaining:
        print(f"\n  Re-run the script to retry {retries_remaining} rate-limited entries.")


def main() -> None:
    parser = argparse.ArgumentParser(description="iTunes album artwork lookup pipeline")
    parser.add_argument("--stats", action="store_true", help="Print cache hit rate and exit")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N pairs (test mode)")
    args = parser.parse_args()

    if args.stats:
        pairs = load_pairs()
        cache = load_cache()
        print_stats(cache, len(pairs))
        return

    run(limit=args.limit)


if __name__ == "__main__":
    main()
