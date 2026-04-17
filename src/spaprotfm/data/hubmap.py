"""HuBMAP CODEX dataset catalog + downloader."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

HUBMAP_SEARCH_URL = "https://search.api.hubmapconsortium.org/v3/portal/search"

# HuBMAP two-letter organ codes for organs of interest
ORGAN_CODES: dict[str, str] = {
    "LARGE_INTESTINE": "LI",
    "SMALL_INTESTINE": "SI",
    "SPLEEN": "SP",
    "LYMPH_NODE": "LY",
}


@dataclass
class HuBMAPDataset:
    hubmap_id: str
    organ: str
    n_markers: int
    download_url: str


def load_catalog(catalog_path: str | Path) -> list[HuBMAPDataset]:
    df = pd.read_csv(catalog_path)
    return [
        HuBMAPDataset(
            hubmap_id=row.hubmap_id,
            organ=row.organ,
            n_markers=int(row.n_markers),
            download_url=str(row.download_url) if pd.notna(row.download_url) else "",
        )
        for row in df.itertuples(index=False)
    ]


def query_hubmap_search(organ: str, assay: str = "CODEX") -> list[dict]:
    """Query HuBMAP search API for public datasets matching organ + assay.

    Args:
        organ: Either a two-letter HuBMAP organ code (e.g. ``"LI"``) or a
               long name key from :data:`ORGAN_CODES` (e.g. ``"LARGE_INTESTINE"``).
        assay: HuBMAP assay type string; defaults to ``"CODEX"``.
    """
    # Accept both long names and raw two-letter codes
    organ_code = ORGAN_CODES.get(organ, organ)
    payload = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"entity_type.keyword": "Dataset"}},
                    {"term": {"data_types.keyword": assay}},
                    {"term": {"origin_samples.organ.keyword": organ_code}},
                    {"term": {"data_access_level": "public"}},
                ]
            }
        },
        "size": 200,
    }
    r = requests.post(HUBMAP_SEARCH_URL, json=payload, timeout=30)
    r.raise_for_status()
    return [hit["_source"] for hit in r.json()["hits"]["hits"]]


def download_dataset(ds: HuBMAPDataset, out_dir: str | Path) -> Path:
    """Download a single dataset to out_dir/<hubmap_id>/. Returns the directory."""
    out = Path(out_dir) / ds.hubmap_id
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", ds.hubmap_id, out)
    return out
