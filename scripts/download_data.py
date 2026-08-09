#!/usr/bin/env python3
"""Fetch the TB chest X-ray dataset into data/raw/.

Primary source is the Kaggle dataset by Tawsifur Rahman:
    https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
(700 TB / 3500 normal films, 512x512 PNG, publicly released for research).

If you have Kaggle credentials, that download is one command:
    kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
and then unzip it so that data/raw/Normal and data/raw/Tuberculosis exist.

For everyone else this script pulls the same images from a public GitHub
mirror (the Kaggle set re-hosted), no credentials needed:

    python scripts/download_data.py                 # full mirror (~380 MB)
    python scripts/download_data.py --normal 1540   # subsample the big class
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://api.github.com/repos/guillecanovas/TuberculosisXRayDetection/tarball/HEAD",
    "https://codeload.github.com/guillecanovas/TuberculosisXRayDetection/tar.gz/HEAD",
    "https://api.github.com/repos/sorna-fast/tb-chest-xray-classifier/tarball/HEAD",
]

UA = {"User-Agent": "tb-detection-data-fetcher"}


def _open_tar(url: str) -> tarfile.TarFile:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=1800) as resp:
        blob = io.BytesIO(resp.read())
    return tarfile.open(fileobj=blob, mode="r:gz")


def download(data_dir: Path, normal_limit: int | None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = ("data/Normal/", "data/Tuberculosis/")

    tf_obj = None
    for url in MIRRORS:
        try:
            print(f"downloading {url} ...")
            tf_obj = _open_tar(url)
            if any(m.name.endswith(".png") or m.name.endswith(".jpg") for m in tf_obj):
                break
        except Exception as exc:  # mirror down/blocked, try next
            print(f"  failed: {exc}")
            tf_obj = None
    if tf_obj is None:
        sys.exit("all mirrors failed - use the Kaggle route (see docstring)")

    kept = []
    for member in tf_obj:
        name = member.name
        if not member.isfile():
            continue
        norm = name.replace("TB_Chest_Radiography_Database/", "data/").split("/", 1)[-1]
        if not any(norm.startswith(w) for w in wanted):
            continue
        member.name = norm[len("data/"):]  # flatten to <Class>/<file>
        kept.append(member)
    if not kept:
        sys.exit("tarball had no dataset images")

    random.Random(42).shuffle(kept)
    counts = {"Normal": 0, "Tuberculosis": 0}
    n = 0
    for member in kept:
        cls = member.name.split("/", 1)[0]
        if cls == "Normal" and normal_limit is not None and counts["Normal"] >= normal_limit:
            continue
        try:
            tf_obj.extract(member, path=data_dir, filter="data")
        except TypeError:  # python < 3.11.4 has no `filter` kwarg, and our names are safe anyway
            tf_obj.extract(member, path=data_dir)
        counts[cls] += 1
        n += 1
        if n % 250 == 0:
            print(f"  {n} images so far...", flush=True)
    print(f"done: {counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    ap.add_argument("--normal", type=int, default=None, help="cap the (much bigger) Normal class")
    args = ap.parse_args()
    download(args.data_dir, args.normal)


if __name__ == "__main__":
    main()
