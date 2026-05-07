#!/usr/bin/env python3
"""
Batch-run Essentia streaming extractor via Docker over a local music library.

Features
- Recursively scans a music root for audio files (mp3/m4a/flac/wav/aiff/aif),
  skipping DRM .m4p files.
- Writes a stable file list (relative and absolute) for reproducibility.
- Runs the Dockerized extractor sequentially or with light parallelism.
- Resumes safely by skipping existing non-empty outputs.

Requirements
- Docker Desktop running
- Image: mtgupf/essentia:latest (pulled automatically on first run by Docker)

Examples
  # Dry run, show what would be processed
  python3 scripts/es_run_extractor.py \
    --music-root "/Users/you/Music/Media.localized/Music" \
    --out-dir data/raw/essentia/extraction --limit 10 --dry-run

  # Process first 100 files sequentially
  python3 scripts/es_run_extractor.py \
    --music-root "/Users/you/Music/Media.localized/Music" \
    --out-dir data/raw/essentia/extraction --limit 100

  # Process with 2 parallel jobs
  python3 scripts/es_run_extractor.py \
    --music-root "/Users/you/Music/Media.localized/Music" \
    --out-dir data/raw/essentia/extraction --jobs 2
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import subprocess


SUPPORTED_EXT = {".mp3", ".m4a", ".flac", ".wav", ".aiff", ".aif"}
SKIP_EXT = {".m4p"}  # DRM


def scan_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, files in os.walk(root):
        d = Path(dirpath)
        for fn in files:
            ext = Path(fn).suffix.lower()
            if ext in SKIP_EXT:
                continue
            if ext in SUPPORTED_EXT:
                out.append(d / fn)
    # deterministic order
    out = sorted(set(p.resolve() for p in out))
    return out


def to_rel_posix(path: Path, start: Path) -> str:
    rel = path.relative_to(start)
    return rel.as_posix()


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def write_lists(rel_list: List[str], abs_list: List[str], out_rel: Path, out_abs: Path):
    ensure_parent(out_rel)
    out_rel.write_text("\n".join(rel_list) + ("\n" if rel_list else ""), encoding="utf-8")
    ensure_parent(out_abs)
    out_abs.write_text("\n".join(abs_list) + ("\n" if abs_list else ""), encoding="utf-8")


def build_docker_cmd(music_root: Path, out_dir: Path, rel_path: str, out_name: str) -> List[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--platform=linux/amd64",
        "-v",
        f"{music_root}:/in:ro",
        "-v",
        f"{out_dir}:/out",
        "mtgupf/essentia:latest",
        "/usr/local/bin/essentia_streaming_extractor_music",
        f"/in/{rel_path}",
        f"/out/{out_name}",
    ]


def process_one(music_root: Path, out_dir: Path, rel_path: str, dry_run: bool = False) -> Tuple[str, str]:
    base = Path(rel_path).name
    out_name = f"{Path(base).stem}-{short_hash(rel_path)}.json"
    out_path = out_dir / out_name
    # resume
    if out_path.exists() and out_path.stat().st_size > 0:
        return rel_path, "skip"
    if dry_run:
        return rel_path, "dry-run"
    cmd = build_docker_cmd(music_root, out_dir, rel_path, out_name)
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode == 0:
            return rel_path, "ok"
        else:
            # write minimal log alongside for inspection
            ensure_parent(out_path)
            with open(out_path.with_suffix(".log"), "w", encoding="utf-8") as f:
                f.write(res.stdout)
            return rel_path, f"fail({res.returncode})"
    except Exception as e:
        return rel_path, f"error({e.__class__.__name__})"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Batch-run Essentia (Docker) over a music library")
    ap.add_argument("--music-root", required=True, help="Root folder of your music library")
    ap.add_argument("--out-dir", default="data/raw/essentia/extraction", help="Output folder for JSON files")
    ap.add_argument("--list-rel", default="data/file_lists/es_all_rel.txt", help="Where to write relative path list")
    ap.add_argument("--list-abs", default="data/file_lists/es_all_abs.txt", help="Where to write absolute path list")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    ap.add_argument("--start", type=int, default=0, help="Start offset within the list")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs (1-3 recommended)")
    ap.add_argument("--dry-run", action="store_true", help="List actions without running Docker")
    args = ap.parse_args(argv)

    music_root = Path(os.path.expanduser(args.music_root)).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scan
    files_abs = scan_files(music_root)
    files_rel = [to_rel_posix(p, music_root) for p in files_abs]

    # Persist lists for reproducibility
    write_lists(files_rel, [str(p) for p in files_abs], Path(args.list_rel), Path(args.list_abs))

    # Slice
    start = max(0, args.start)
    end = start + args.limit if args.limit else None
    batch = files_rel[start:end]

    if not batch:
        print("No files to process (check music root or filters)")
        return 0

    print({
        "music_root": str(music_root),
        "out_dir": str(out_dir),
        "total_found": len(files_rel),
        "processing": len(batch),
        "start": start,
        "limit": args.limit,
        "jobs": args.jobs,
        "dry_run": args.dry_run,
    })

    # Execute
    results: List[Tuple[str, str]] = []
    if args.jobs <= 1:
        for rel in batch:
            rel, status = process_one(music_root, out_dir, rel, dry_run=args.dry_run)
            print(f"{status}: {rel}")
            results.append((rel, status))
    else:
        # Light parallelism
        with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = [ex.submit(process_one, music_root, out_dir, rel, args.dry_run) for rel in batch]
            for fut in cf.as_completed(futs):
                rel, status = fut.result()
                print(f"{status}: {rel}")
                results.append((rel, status))

    ok = sum(1 for _, s in results if s == "ok")
    skip = sum(1 for _, s in results if s == "skip")
    fail = len(results) - ok - skip - sum(1 for _, s in results if s == "dry-run")
    print({"ok": ok, "skip": skip, "fail": fail})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

