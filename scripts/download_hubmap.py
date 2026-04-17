"""Download HuBMAP CODEX datasets listed in the catalog.

Usage:
    # Step 1: refresh catalog from API
    uv run python scripts/download_hubmap.py \
        --catalog data/manifests/hubmap_codex.csv \
        --out data/raw/hubmap \
        --refresh-catalog

    # Step 2: download (after manually filling download_url column for desired rows)
    uv run python scripts/download_hubmap.py \
        --catalog data/manifests/hubmap_codex.csv \
        --out data/raw/hubmap \
        --organ LARGE_INTESTINE --limit 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from spaprotfm.data.hubmap import load_catalog, query_hubmap_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_url(url: str, dest: Path, chunk: int = 1024 * 1024) -> None:
    if dest.exists():
        log.info("Skip existing: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for buf in r.iter_content(chunk):
                f.write(buf)
                pbar.update(len(buf))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--organ", default=None, help="Filter by organ (optional)")
    p.add_argument("--limit", type=int, default=0, help="Limit number of datasets (0 = all)")
    p.add_argument("--refresh-catalog", action="store_true",
                   help="Re-query HuBMAP API and overwrite catalog")
    args = p.parse_args()

    if args.refresh_catalog:
        rows = []
        for organ in ["LARGE_INTESTINE", "SMALL_INTESTINE", "SPLEEN", "LYMPH_NODE"]:
            log.info("Querying HuBMAP for organ=%s ...", organ)
            try:
                hits = query_hubmap_search(organ)
            except Exception as e:
                log.warning("Query failed for %s: %s", organ, e)
                continue
            log.info("  got %d hits", len(hits))
            for h in hits:
                meta = h.get("metadata", {})
                # number_of_antibodies is the reliable field; fall back to len(antibodies list)
                n_ab = meta.get("number_of_antibodies") or len(meta.get("antibodies", [])) or -1
                rows.append({
                    "hubmap_id": h.get("hubmap_id"),
                    "organ": organ,
                    "n_markers": n_ab,
                    "assay": "CODEX",
                    "uuid": h.get("uuid"),
                    "download_url": "",
                })
        args.catalog.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.catalog, index=False)
        log.info("Wrote %d rows to %s", len(rows), args.catalog)
        return 0

    catalog = load_catalog(args.catalog)
    if args.organ:
        catalog = [c for c in catalog if c.organ == args.organ]
    if args.limit:
        catalog = catalog[: args.limit]

    log.info("Will download %d datasets to %s", len(catalog), args.out)
    for ds in catalog:
        if not ds.download_url:
            log.warning("No download_url for %s, skipping", ds.hubmap_id)
            continue
        fetch_url(ds.download_url, args.out / ds.hubmap_id / Path(ds.download_url).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
