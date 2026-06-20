#!/usr/bin/env python
"""Download + extract + sample AnghaBench (~1M compilable C functions).

AnghaBench is distributed as a .tar.gz of one-function-per-file .c sources,
mined from public C repos and made standalone-compilable via type inference.
That is exactly the corpus our IR step wants (each .c compiles in isolation).

This script:
  1. downloads the fixed GitHub archive tarball (pinned commit + sha256),
  2. extracts it,
  3. optionally SAMPLES N .c files into a flat directory for a quick first run.

Usage:
    # download + extract full corpus, then sample 10k files for a fast run
    python scripts/fetch_anghabench.py --out data/anghabench --sample 10000

    # just download + extract everything (no sampling)
    python scripts/fetch_anghabench.py --out data/anghabench

The sampled directory is what you point `train.py --sources` at.

Notes:
  * The full archive is large; on Dalia run this on a compute node into $WORK
    (HOME is only 3 GB). Use --no-download if the archive is already present.
  * Sampling copies files (small) so the original extraction can be deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ARCHIVE_URL = (
    "https://github.com/brenocfg/AnghaBench/archive/"
    "d8034ac8562b8c978376008f4b33df01b8887b19.tar.gz"
)
ARCHIVE_SHA256 = "85d068e4ce44f2581e3355ee7a8f3ccb92568e9f5bd338bc3a918566f3aff42f"
EXTRACT_PREFIX = "AnghaBench-d8034ac8562b8c978376008f4b33df01b8887b19"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(out_dir: Path, *, verify: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "anghabench.tar.gz"
    if tar_path.exists():
        print(f"[fetch] archive already present: {tar_path}")
    else:
        print(f"[fetch] downloading {ARCHIVE_URL}")
        print("        (this is large; be patient)")
        with urllib.request.urlopen(ARCHIVE_URL) as resp, open(tar_path, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
    if verify:
        print("[fetch] verifying sha256 ...")
        got = _sha256(tar_path)
        if got != ARCHIVE_SHA256:
            raise SystemExit(
                f"sha256 mismatch!\n  expected {ARCHIVE_SHA256}\n  got      {got}"
            )
        print("[fetch] sha256 OK")
    return tar_path


def extract(tar_path: Path, out_dir: Path) -> Path:
    target = out_dir / EXTRACT_PREFIX
    if target.exists():
        print(f"[fetch] already extracted: {target}")
        return target
    print(f"[fetch] extracting {tar_path} -> {out_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        # safe extraction (avoid path traversal)
        members = []
        for m in tf.getmembers():
            mp = (out_dir / m.name).resolve()
            if not str(mp).startswith(str(out_dir.resolve())):
                raise SystemExit(f"unsafe path in archive: {m.name}")
            members.append(m)
        tf.extractall(out_dir, members=members)
    return target


def sample(src_root: Path, dst: Path, n: int, seed: int = 0) -> int:
    """Copy n random .c files from src_root into a flat dst dir."""
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] scanning .c files under {src_root} ...")
    all_c = list(src_root.rglob("*.c"))
    print(f"[fetch] found {len(all_c):,} .c files")
    rng = random.Random(seed)
    chosen = all_c if n >= len(all_c) else rng.sample(all_c, n)
    copied = 0
    for i, p in enumerate(chosen):
        # flatten name; prefix with index to avoid collisions
        out = dst / f"{i:07d}_{p.name}"
        try:
            shutil.copyfile(p, out)
            copied += 1
        except OSError:
            pass
    print(f"[fetch] sampled {copied:,} files -> {dst}")
    return copied


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch + sample AnghaBench")
    p.add_argument("--out", default="data/anghabench", help="working dir")
    p.add_argument("--sample", type=int, default=None,
                   help="copy N random .c into <out>/sample for a quick run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-download", action="store_true",
                   help="skip download (archive/extraction already present)")
    p.add_argument("--no-verify", action="store_true", help="skip sha256 check")
    args = p.parse_args()

    out_dir = Path(args.out)
    if args.no_download:
        extracted = out_dir / EXTRACT_PREFIX
        if not extracted.exists():
            raise SystemExit(f"--no-download but {extracted} missing")
    else:
        tar = download(out_dir, verify=not args.no_verify)
        extracted = extract(tar, out_dir)

    if args.sample is not None:
        n = sample(extracted, out_dir / "sample", args.sample, seed=args.seed)
        print(f"\n[fetch] DONE. Train on the sample with:\n"
              f"  python scripts/train.py --sources {out_dir/'sample'} --glob '*.c'")
    else:
        print(f"\n[fetch] DONE. Full corpus at {extracted}\n"
              f"  python scripts/train.py --sources {extracted} --glob '**/*.c'")


if __name__ == "__main__":
    main()