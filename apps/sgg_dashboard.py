"""
SGG Dashboard — Soapy GrooveGraph Streamlit UI
===============================================
Three panels:
  1. RAG Query       — natural language question → LLM cited answer → hero result + supporting list
  2. Audio Similarity — pick a track/artist → top-k nearest neighbors by 12-dim audio vector
  3. Feature Inspector — pick any track → raw Essentia feature values

Backends:
  - Qdrant sgg_text_v1  (768-dim text embeddings, nomic-embed-text via Ollama)
  - Qdrant sgg_audio_v1 (12-dim z-scored audio vectors)
  - DuckDB fct_audio_vector_v1 (source of truth for track metadata and audio features)
  - Ollama gemma3:27b for RAG answer generation
  - iTunes image cache (data/cache/itunes_image_cache.json) for album art

Run from repo root:
  streamlit run apps/sgg_dashboard.py

Change Log
----------
2026-04-29 — Score-weighted random sampling in query_text_collection
  Problem: The "Ask Your Library" tab kept recommending the same small set of tracks
  regardless of what the user searched for. This is a known vector search issue called
  "hubness" — certain tracks end up near the center of the embedding space and become
  nearest neighbors for almost every query, even when they aren't a good fit.

  Fix: Instead of always returning the top-K results, the function now fetches a larger
  candidate pool (up to 5x the requested number, capped at 50), then uses score-weighted
  random sampling to pick the final results. Tracks with higher similarity scores are more
  likely to be selected, but the outcome is no longer deterministic — so hub tracks no
  longer lock up every result slot on every query. Results are re-sorted by score before
  display so the best matches still appear first.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import duckdb
import requests
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from itunes_image import load_cache, get_artwork_url, resolve_hero

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DUCKDB_PATH = Path(__file__).parents[1] / "dbt_sgg" / "sgg_prod.duckdb"
VIEW = "fct_audio_vector_v1"
TEXT_COLLECTION = "sgg_text_v1"
AUDIO_COLLECTION = "sgg_audio_v1"
AUDIO_DIM = 12
DEFAULT_TOPK = 5

VECTOR_ORDER = [
    "danceability", "bpm", "loudness_integrated", "loudness_range",
    "spectral_rms", "spectral_centroid", "spectral_flux", "pitch_salience",
    "hfc", "zcr", "onset_rate", "key_strength",
]

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_qdrant() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6335")
    return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)


@st.cache_resource
def get_duckdb():
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


@st.cache_data
def load_image_cache() -> dict:
    return load_cache()


@st.cache_data
def load_track_list() -> list[dict]:
    con = get_duckdb()
    rows = con.execute(
        f"select file_hash, artist, album, title, date, genre, key_key, key_scale, bpm, danceability, loudness_integrated "
        f"from {VIEW} order by artist, title"
    ).fetch_df().to_dict(orient="records")
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ollama_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL_HOST", "http://localhost:11434")


def embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def chat_model() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "gemma3:27b")


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{ollama_url()}/api/embeddings",
        json={"model": embed_model(), "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _weighted_sample(points: list, k: int) -> list:
    """Sample k points without replacement, weighted by score."""
    pool = list(points)
    selected = []
    for _ in range(min(k, len(pool))):
        min_score = min(p.score for p in pool)
        weights = [p.score - min_score + 0.01 for p in pool]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0.0
        for i, (w, point) in enumerate(zip(weights, pool)):
            cumulative += w
            if cumulative >= r:
                selected.append(point)
                pool.pop(i)
                break
    return selected


def query_text_collection(vector: list[float], topk: int) -> list[dict]:
    client = get_qdrant()
    # Over-fetch so we have a candidate pool to sample from, reducing hub dominance
    fetch_k = min(topk * 5, 50)
    results = client.query_points(
        collection_name=TEXT_COLLECTION,
        query=vector,
        limit=fetch_k,
        with_payload=True,
        with_vectors=False,
    ).points

    sampled = _weighted_sample(results, topk)
    sampled.sort(key=lambda p: p.score, reverse=True)

    return [
        {
            "score": round(p.score, 4),
            "file_hash": p.payload.get("file_hash"),
            "artist": p.payload.get("artist"),
            "album": p.payload.get("album"),
            "title": p.payload.get("title"),
            "date": str(p.payload.get("date") or ""),
            "genre": p.payload.get("genre"),
            "key_key": p.payload.get("key_key"),
            "key_scale": p.payload.get("key_scale"),
            "doc": p.payload.get("doc"),
        }
        for p in sampled
    ]


def generate_similarity_comment(seed: dict, results: list[dict]) -> str:
    seed_str = f"{seed.get('artist')} — {seed.get('title')} ({seed.get('album')})"
    neighbor_lines = []
    for i, r in enumerate(results, start=1):
        key = f"{r.get('key_key', '')} {r.get('key_scale', '')}".strip()
        neighbor_lines.append(
            f"{i}. {r.get('artist')} — {r.get('title')} ({r.get('album')}) "
            f"[genre: {r.get('genre') or '—'}, key: {key}, score: {r.get('score')}]"
        )
    neighbors_str = "\n".join(neighbor_lines)
    prompt = (
        f"You are a music expert. A user is exploring tracks similar to:\n\n"
        f"  {seed_str}\n\n"
        f"The following tracks were found as the closest audio matches by features like BPM, key, energy, and danceability:\n\n"
        f"{neighbors_str}\n\n"
        f"Write 2-3 sentences explaining what these tracks likely have in common with the seed track. "
        f"Be specific — mention genre, mood, energy, or tempo where relevant. "
        f"Cite at least two of the neighbor tracks by artist and title. "
        f"Do not invent tracks that aren't listed."
    )
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
    return resp.json()["message"]["content"]


def generate_rag_answer(query: str, results: list[dict]) -> str:
    context_lines = []
    for i, r in enumerate(results, start=1):
        context_lines.append(
            f"{i}. {r['artist']} — {r['title']} ({r['album']})\n   {r.get('doc', '')}"
        )
    context = "\n".join(context_lines)
    prompt = (
        f"You are a music recommendation assistant. A user asked: \"{query}\"\n\n"
        f"Here are the most relevant tracks from their personal music library:\n\n"
        f"{context}\n\n"
        f"Based only on the tracks above, write a short, friendly recommendation. "
        f"Cite specific tracks by artist and title. Do not invent tracks that aren't listed."
    )
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
    return resp.json()["message"]["content"]


def query_audio_collection(file_hash: str, topk: int, exclude_album: str | None = None) -> list[dict]:
    con = get_duckdb()
    rows = con.execute(
        f"select file_hash, vector_z, artist, album, title, date, genre, key_key, key_scale, bpm, danceability, loudness_integrated "
        f"from {VIEW} where file_hash = ? limit 1",
        [file_hash],
    ).fetch_df().to_dict(orient="records")

    if not rows:
        return []

    row = rows[0]
    vcol = row.get("vector_z")
    if vcol is None:
        return []
    try:
        vector = [float(x) for x in list(vcol)]
    except (TypeError, ValueError):
        return []

    if len(vector) != AUDIO_DIM:
        return []

    client = get_qdrant()
    results = client.search(
        collection_name=AUDIO_COLLECTION,
        query_vector=vector,
        limit=(topk * 3) + 1,  # over-fetch to absorb same-album and seed exclusions
        with_payload=True,
        with_vectors=False,
    )
    return [
        {
            "score": round(p.score, 4),
            "file_hash": p.payload.get("file_hash"),
            "artist": p.payload.get("artist"),
            "album": p.payload.get("album"),
            "title": p.payload.get("title"),
            "date": str(p.payload.get("date") or ""),
            "genre": p.payload.get("genre"),
            "key_key": p.payload.get("key_key"),
            "key_scale": p.payload.get("key_scale"),
        }
        for p in results
        if p.payload.get("file_hash") != file_hash
        and (exclude_album is None or p.payload.get("album") != exclude_album)
    ][:topk]


def render_track_card(result: dict, cache: dict, show_score: bool = False) -> None:
    artist = result.get("artist", "")
    album = result.get("album", "")
    title = result.get("title", "")
    year = str(result.get("date") or "").split("-")[0] or "—"
    genre = result.get("genre") or "—"
    key = f"{result.get('key_key', '')} {result.get('key_scale', '')}".strip() or "—"
    artwork_url = result.get("artwork_url") or get_artwork_url(artist, album, cache)

    col_img, col_text = st.columns([1, 3])
    with col_img:
        if artwork_url:
            st.image(artwork_url, width=120)
        else:
            st.markdown("🎵")
    with col_text:
        st.markdown(f"**{title}**")
        st.markdown(f"{artist} · *{album}* ({year})")
        st.markdown(f"Genre: {genre} · Key: {key}")
        if show_score:
            st.caption(f"Score: {result.get('score', '')}")


# ---------------------------------------------------------------------------
# Panel: RAG Query
# ---------------------------------------------------------------------------

def panel_rag() -> None:
    st.header("Ask Your Library")
    st.markdown("<p style='font-size: 1.1rem; color: #555;'>Ask a question and receive recommendations from the music library.</p>", unsafe_allow_html=True)

    query = st.text_input("What are you in the mood for?", placeholder="e.g. slow jazzy late night music")
    topk = st.slider("Tracks to retrieve", min_value=3, max_value=10, value=DEFAULT_TOPK)

    if st.button("Search", key="rag_search") and query.strip():
        cache = load_image_cache()

        with st.spinner("Embedding query..."):
            try:
                vector = get_embedding(query)
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                return

        with st.spinner("Retrieving tracks..."):
            results = query_text_collection(vector, topk)

        if not results:
            st.warning("No results returned.")
            return

        with st.spinner("Asking the Clanker to dig through this pseudo-hippie music collection..."):
            try:
                answer = generate_rag_answer(query, results)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                return

        st.markdown("### Recommendation")
        st.markdown(answer)

        hero, supporting = resolve_hero(results, cache)

        st.markdown("---")
        st.markdown("### Top Pick")
        render_track_card(hero, cache)

        if supporting:
            st.markdown("---")
            st.markdown("### Also Consider")
            for r in supporting[:5]:
                render_track_card(r, cache)
                st.markdown("&nbsp;")


# ---------------------------------------------------------------------------
# Shared: cascading Artist → Album → Track selection
# ---------------------------------------------------------------------------

def cascading_track_select(tracks: list[dict], key_prefix: str) -> dict | None:
    """Render three cascading selectboxes and return the selected track dict."""
    artists = sorted({r["artist"] for r in tracks if r.get("artist")})
    selected_artist = st.selectbox("Artist", options=artists, key=f"{key_prefix}_artist")

    albums = sorted({r["album"] for r in tracks if r.get("artist") == selected_artist and r.get("album")})
    selected_album = st.selectbox("Album", options=albums, key=f"{key_prefix}_album")

    titles = [
        r for r in tracks
        if r.get("artist") == selected_artist and r.get("album") == selected_album
    ]
    if not titles:
        return None

    title_labels = [r["title"] or "Unknown" for r in titles]
    selected_title = st.selectbox("Track", options=title_labels, key=f"{key_prefix}_track")
    idx = title_labels.index(selected_title)
    return titles[idx]


# ---------------------------------------------------------------------------
# Panel: Audio Similarity
# ---------------------------------------------------------------------------

def panel_audio_similarity() -> None:
    st.header("Audio Similarity Search")
    st.markdown("<p style='font-size: 1.1rem; color: #555;'>Pick a track and find the closest matches by audio features (BPM, key, energy, danceability).</p>", unsafe_allow_html=True)

    tracks = load_track_list()
    cache = load_image_cache()

    # Random Track button — pre-populates the cascading dropdowns and auto-runs
    if st.button("Random Track", key="audio_random"):
        pick = random.choice(tracks)
        st.session_state["audio_artist"] = pick["artist"]
        st.session_state["audio_album"] = pick["album"]
        st.session_state["audio_track"] = pick["title"] or "Unknown"
        st.session_state["audio_auto_run"] = True

    seed = cascading_track_select(tracks, key_prefix="audio")
    topk = st.slider("Similar tracks to show", min_value=3, max_value=10, value=DEFAULT_TOPK, key="audio_topk")

    auto_run = st.session_state.pop("audio_auto_run", False)

    if (st.button("Find Similar Tracks", key="audio_search") or auto_run) and seed:
        with st.spinner("Querying Qdrant audio collection..."):
            results = query_audio_collection(seed["file_hash"], topk, exclude_album=seed.get("album"))

        if not results:
            st.warning("No results — check that sgg_audio_v1 is populated and vector_z is present.")
            return

        st.markdown(f"### Selected Track: {seed['artist']} — {seed['title']}")
        render_track_card(
            {**seed, "date": str(seed.get("date") or ""), "key_key": seed.get("key_key"), "key_scale": seed.get("key_scale")},
            cache,
        )

        with st.spinner("Clanker is vibing on these results, hold tight..."):
            try:
                comment = generate_similarity_comment(seed, results)
                st.markdown("### What They Have in Common")
                st.markdown(comment)
            except Exception as e:
                st.caption(f"Commentary unavailable: {e}")

        st.markdown("---")
        st.markdown("### Similar Tracks")
        for r in results:
            render_track_card(r, cache, show_score=True)
            st.markdown("&nbsp;")


# ---------------------------------------------------------------------------
# Panel: Feature Inspector
# ---------------------------------------------------------------------------

def panel_feature_inspector() -> None:
    st.header("Feature Inspector")
    st.markdown("<p style='font-size: 1.1rem; color: #555;'>Inspect the raw Essentia audio features for any track in the library.</p>", unsafe_allow_html=True)

    tracks = load_track_list()
    row = cascading_track_select(tracks, key_prefix="inspector")
    if not row:
        st.warning("No tracks found for the selected artist and album.")
        return

    cache = load_image_cache()
    artist = row.get("artist", "")
    album = row.get("album", "")
    year = str(row.get("date") or "").split("-")[0] or "—"
    artwork_url = get_artwork_url(artist, album, cache)

    col_img, col_meta = st.columns([1, 3])
    with col_img:
        if artwork_url:
            st.image(artwork_url, width=120)
        else:
            st.markdown("🎵")
    with col_meta:
        st.markdown(f"**{row.get('title', '—')}**")
        st.markdown(f"{artist} · *{album}* ({year})")
        st.markdown(f"Genre: {row.get('genre') or '—'}")
        key = f"{row.get('key_key', '')} {row.get('key_scale', '')}".strip() or "—"
        st.markdown(f"Key: {key}")

    st.markdown("---")
    st.markdown("### Audio Features")

    feature_labels = {
        "bpm": "BPM",
        "danceability": "Danceability",
        "loudness_integrated": "Loudness (LUFS)",
        "key_key": "Key",
        "key_scale": "Scale",
    }

    cols = st.columns(len(feature_labels))
    for col, (field, label) in zip(cols, feature_labels.items()):
        val = row.get(field)
        display = f"{val:.1f}" if isinstance(val, float) else (str(val) if val else "—")
        col.metric(label, display)


# ---------------------------------------------------------------------------
# Panel: About
# ---------------------------------------------------------------------------

def panel_about() -> None:
    st.header("About Soapy GrooveGraph")
    st.markdown("""
---

### What Is This?

Soapy GrooveGraph is an AI-powered music discovery tool built on top of Matt's personal iTunes library.
This is not a streaming service. Nor a recommendation algorithm trained on 100 million users.
Just 9,294 songs, a local AI, and the audacity to think this app should be created at all - much less for "fun."

### Why Does This Exist?

Because Spotify's algorithm doesn't really know you — it knows the aggregate of everyone who listens to the same songs as you. That's not the same thing.

SGG knows *your* library. Every obscure album you imported from a CD in 2003. Every artist no one else is familiar with.
Every genre mislabeled as "Other." It finds patterns in your actual collection and surfaces music you already own, but might have forgotten about.

---

### The Objective

To answer the age-old question: *"What do I feel like listening to right now?"* — without handing your music taste
over to a Silicon Valley algorithm that thinks you want to hear the same three artists on repeat forever.

SGG lets you ask questions about your own music collection and actually get useful (mostly) answers back.
It also finds tracks that sound similar to each other based on real audio features — not just "people who liked this also liked that."

---

### The Data

**9,294 tracks. 558 artists. 841 albums.**

All music is extracted from a personal iTunes library using Essentia, an open-source audio analysis tool that listens to
every song and produces a detailed fingerprint: BPM, musical key, energy, danceability, mood, and more.

No Spotify. No streaming APIs. No data brokers. Just the actual audio files sitting on a hard drive, finally being put to work. 
I know, I know: there are some questionable genre classifications in here. But give Essentia a break, it did its best and it's free and open source. 

---

### The Tools

| What | Why |
|---|---|
| **Essentia** | Extracts audio features from every track — BPM, key, mood, energy, danceability |
| **dbt + DuckDB** | Cleans, deduplicates, and transforms the raw features into something usable |
| **Qdrant** | Vector database that stores the audio fingerprints and text embeddings for similarity search |
| **Ollama + nomic-embed-text** | Turns track descriptions into 768-dimensional vectors. Locally. On your own machine. |
| **Ollama + gemma3:27b** | The LLM writing those recommendation paragraphs. Also local. Also on your machine. |
| **Streamlit** | The thing you're looking at right now |
| **iTunes Search API** | Free Apple API used to fetch album art. Surprisingly useful. 75% hit rate. |

---

### The Fine Print

- All AI runs **locally** — nothing leaves your machine.
- Album art is fetched from the iTunes Search API at setup time and cached — no live API calls during use.
- The dog with antlers is named Soapy. He is not available for licensing.
""")


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

TAB_STYLES = """
<style>
/* Tab bar — more breathing room */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0;
}
/* Every tab — pill shape, larger text */
.stTabs [data-baseweb="tab"] {
    height: 48px;
    padding: 0 28px;
    border-radius: 6px 6px 0 0;
    font-size: 16px;
    font-weight: 600;
    color: #555;
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-bottom: none;
}
/* Active tab */
.stTabs [aria-selected="true"] {
    background-color: #ff4b4b !important;
    color: white !important;
    border-color: #ff4b4b !important;
}
/* Hover on inactive tabs */
.stTabs [data-baseweb="tab"]:hover {
    background-color: #ffe5e5;
    color: #ff4b4b;
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Soapy GrooveGraph", page_icon="🎵", layout="wide")
    st.markdown(TAB_STYLES, unsafe_allow_html=True)
    col_img, col_title = st.columns([1, 3])
    with col_img:
        st.image(str(Path(__file__).parent / "images" / "soapy_sunglasses_bw.jpeg"), width=280)
    with col_title:
        st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0;'>Soapy GrooveGraph</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.1rem; color: #555;'>Your personal music library, powered by AI — discover tracks, explore sounds, and find your next favorite song.</p>", unsafe_allow_html=True)

    tab_rag, tab_audio, tab_inspector, tab_about = st.tabs(["Ask Your Library", "Audio Similarity", "Feature Inspector", "About"])

    with tab_rag:
        panel_rag()

    with tab_audio:
        panel_audio_similarity()

    with tab_inspector:
        panel_feature_inspector()

    with tab_about:
        panel_about()


if __name__ == "__main__":
    main()
