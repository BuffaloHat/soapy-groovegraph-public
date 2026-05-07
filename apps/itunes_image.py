"""
iTunes Search API image lookup utility.
Loads the pre-built cache from data/cache/itunes_image_cache.json.
Used by the Streamlit app — never calls the API at query time.
In short: scripts/itunes_image_lookup.py built the cache, 
and this file is how the Streamlit app uses it.
"""

import json
from pathlib import Path
from typing import Optional

DEFAULT_CACHE_PATH = Path(__file__).parents[1] / "data" / "cache" / "itunes_image_cache.json"


def _cache_key(artist: str, album: str) -> str:
    return f"{artist.lower().strip()}|{album.lower().strip()}"


def load_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> dict:
    if not cache_path.exists():
        return {}
    with open(cache_path) as f:
        return json.load(f)


def get_artwork_url(artist: str, album: str, cache: dict) -> Optional[str]:
    return cache.get(_cache_key(artist, album))


def resolve_hero(results: list, cache: dict) -> tuple:
    """
    Given a list of top-k result dicts (each with 'artist' and 'album' keys),
    returns (hero, remaining) where hero is the first result with a valid
    cached image and remaining is the rest in original order (hero excluded).
    Falls back to results[0] as hero with no image if none have art.
    """
    hero = None
    hero_idx = None

    for i, result in enumerate(results):
        url = get_artwork_url(result.get("artist", ""), result.get("album", ""), cache)
        if url:
            hero = {**result, "artwork_url": url}
            hero_idx = i
            break

    if hero is None and results:
        hero = {**results[0], "artwork_url": None}
        hero_idx = 0

    remaining = [r for i, r in enumerate(results) if i != hero_idx]
    return hero, remaining
