#!/usr/bin/env python3
"""
Build a manifest of iTunes audio files for Essentia extraction.

Scans a music library root (Apple Music layout) and outputs file path lists
used as input to es_run_extractor.py. Supports full-library scans or
targeted subsets by artist prefix.

Outputs:
  - data/file_lists/es_targets.txt       (relative paths, one per line)
  - data/file_lists/es_targets_abs.txt   (absolute paths; informational)
  - data/file_lists/es_match_report.csv  (match audit)

Usage example:
  python3 scripts/es_select_files.py \
    --music-root "/Users/you/Music/Media.localized/Music"

  # Filter to a single artist prefix
  python3 scripts/es_select_files.py \
    --music-root "/Users/you/Music/Media.localized/Music" \
    --artist-prefix "B" --lenient
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple
import unicodedata
import re


SUPPORTED_EXT = {".mp3", ".m4a", ".flac", ".wav", ".aiff", ".aif"}
SKIP_EXT = {".m4p"}  # DRM


def collapse_ws(s: str) -> str:
    return " ".join(str(s).strip().split())


def strip_feat_suffix(s: str) -> str:
    import re

    return re.sub(r"\s+(feat\.?|featuring|ft\.)\s+.*$", "", str(s), flags=re.IGNORECASE)


def to_candidate(s: str) -> str:
    return collapse_ws(strip_feat_suffix(s)).lower()


def ascii_fold_lower(s: str) -> str:
    """Best-effort ASCII fold for matching (remove diacritics)."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def lenient_norm(s: str) -> str:
    """Lenient normalization to match filesystem artist directories to MB names.

    - ASCII-fold and lowercase
    - Replace common separators (/_.,&,+) with spaces; map '&' to 'and'
    - Remove any non-alphanumeric
    - Collapse whitespace
    - Strip common trailing artifacts like underscores
    """
    s = ascii_fold_lower(s)
    s = s.replace("&", " and ")
    s = s.replace("+", " and ")
    s = re.sub(r"[\./:_-]+", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = collapse_ws(s)
    return s


def load_target_artists(path: str, use_lenient: bool) -> Set[str]:
    """Load resolved artists from artist_map_final.csv and return normalized keys.

    Accept both artist_raw and canonical_name when available.
    Only include rows that have an mbid (resolved), so we target known artists.
    """
    targets: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            mbid = (row.get("mbid") or "").strip()
            if not mbid:
                continue
            for key in (row.get("artist_raw"), row.get("canonical_name")):
                if not key:
                    continue
                cand = lenient_norm(key) if use_lenient else to_candidate(key)
                if cand:
                    targets.add(cand)
    return targets


def scan_artists(music_root: str) -> List[Tuple[str, str]]:
    """Return list of (artist_dir_name, artist_abs_path) under music_root (depth=1)."""
    out: List[Tuple[str, str]] = []
    try:
        for name in os.listdir(music_root):
            p = os.path.join(music_root, name)
            if os.path.isdir(p):
                out.append((name, p))
    except FileNotFoundError:
        raise SystemExit(f"Music root not found: {music_root}")
    return out


def collect_files(artist_dir: str) -> Iterable[str]:
    """Yield absolute file paths recursively for supported extensions (skip DRM)."""
    for root, _, files in os.walk(artist_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT:
                continue
            if ext not in SUPPORTED_EXT:
                continue
            yield os.path.join(root, fn)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select audio files for Essentia by MB artists")
    ap.add_argument("--music-root", required=True, help="Root folder of your music library")
    ap.add_argument(
        "--artists-csv",
        default="data/raw/musicbrainz/resolution/artist_map_final.csv",
        help="CSV with columns including artist_raw, canonical_name, mbid",
    )
    ap.add_argument(
        "--artist-prefix",
        default=None,
        help="Optional case-insensitive prefix filter on artist directory name (e.g., 'A')",
    )
    ap.add_argument(
        "--out-list",
        default="data/file_lists/es_targets.txt",
        help="Output text file with RELATIVE paths (one per line)",
    )
    ap.add_argument(
        "--out-list-abs",
        default="data/file_lists/es_targets_abs.txt",
        help="Output text file with ABSOLUTE paths (informational)",
    )
    ap.add_argument(
        "--lenient",
        action="store_true",
        help="Use lenient matching (punctuation/diacritics/&_ variations)",
    )
    ap.add_argument(
        "--debug-report",
        default="data/file_lists/es_match_report.csv",
        help="Write a CSV report of artist directory matches",
    )
    args = ap.parse_args(argv)

    music_root = os.path.abspath(os.path.expanduser(args.music_root))

    targets = load_target_artists(args.artists_csv, use_lenient=args.lenient)
    if not targets:
        print("No target artists loaded (mbid-filtered); check input CSV", file=sys.stderr)

    artists = scan_artists(music_root)
    # Build normalized map of local artist dirs -> absolute path
    if args.lenient:
        norm_to_path: Dict[str, str] = {lenient_norm(name): path for name, path in artists}
    else:
        norm_to_path = {to_candidate(name): path for name, path in artists}
    matched_keys = sorted(set(norm_to_path.keys()) & targets)
    # Optional prefix filter on normalized artist directory names
    if args.artist_prefix:
        pref = str(args.artist_prefix).lower()
        matched_keys = [k for k in matched_keys if k.startswith(pref)]

    selected_abs: List[str] = []
    # Build debug report data
    report_rows: List[Tuple[str, str, bool, str, int]] = []
    # Map back from norm key to original dir name (first occurrence)
    norm_to_dirname: Dict[str, str] = {}
    for name, path in artists:
        k = lenient_norm(name) if args.lenient else to_candidate(name)
        if k not in norm_to_dirname:
            norm_to_dirname[k] = name
    matched_set = set(matched_keys)
    for k, p in norm_to_path.items():
        files_here = list(collect_files(p))
        if k in matched_set:
            selected_abs.extend(files_here)
            report_rows.append((norm_to_dirname.get(k, k), k, True, k, len(files_here)))
        else:
            # Only include unmatched when prefix filter is active (focused review)
            if args.artist_prefix:
                report_rows.append((norm_to_dirname.get(k, k), k, False, "", len(files_here)))

    # Ensure deterministic order
    selected_abs = sorted(set(selected_abs))

    # Write outputs
    os.makedirs(os.path.dirname(args.out_list), exist_ok=True)
    with open(args.out_list_abs, "w", encoding="utf-8") as fa, open(
        args.out_list, "w", encoding="utf-8"
    ) as fr:
        for fp in selected_abs:
            fa.write(fp + "\n")
            # Relative to music_root for easier docker mounts
            rel = os.path.relpath(fp, start=music_root)
            # Always use forward slashes in the list for docker clarity
            rel = rel.replace(os.sep, "/")
            fr.write(rel + "\n")

    # Write debug report
    try:
        with open(args.debug_report, "w", encoding="utf-8", newline="") as rep:
            rep.write("artist_dir,normalized_key,matched,matched_key,files_in_dir\n")
            for row in report_rows:
                artist_dir, norm_key, matched, matched_key, count = row
                rep.write(f"{artist_dir},{norm_key},{str(matched).lower()},{matched_key},{count}\n")
    except Exception:
        pass

    print(
        {
            "music_root": music_root,
            "artist_dirs_total": len(artists),
            "artists_matched": len(matched_keys),
            "files_selected": len(selected_abs),
            "out_list": args.out_list,
            "out_list_abs": args.out_list_abs,
            "report": args.debug_report,
            "lenient": args.lenient,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
