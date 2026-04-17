"""Marker name standardization across CODEX/IMC datasets."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_GREEK_TO_LATIN = {"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "κ": "k"}
_DASH_LIKE = re.compile(r"[\s_]+")


def canonicalize(name: str) -> str:
    """Normalize a marker name to lowercase ASCII with hyphen separators."""
    s = name.strip()
    for greek, latin in _GREEK_TO_LATIN.items():
        s = s.replace(greek, latin)
    # Decompose unicode and drop diacritics
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = _DASH_LIKE.sub("-", s)
    return s.lower()


def load_alias_table(path: str | Path) -> dict[str, str]:
    """Load YAML mapping {canonical: [aliases]} and invert to {alias_canon: canonical}."""
    raw = yaml.safe_load(Path(path).read_text())
    table: dict[str, str] = {}
    for canonical, aliases in raw.items():
        names = [canonical, *aliases]
        for name in names:
            table[canonicalize(name)] = canonical
    return table


def standardize_panel(
    names: list[str], alias_table: dict[str, str]
) -> list[str | None]:
    """Map raw marker names to canonical names; return None for unknown."""
    out: list[str | None] = []
    for n in names:
        key = canonicalize(n)
        if key in alias_table:
            out.append(alias_table[key])
        else:
            logger.warning("Unknown marker: %r (canonical: %r)", n, key)
            out.append(None)
    return out
