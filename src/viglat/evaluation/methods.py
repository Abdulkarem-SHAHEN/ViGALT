"""Canonical method identifiers for ViGALT vs CAID evaluation."""

from __future__ import annotations

METHOD_VIGALT = "ViGALT"
METHOD_CAID = "CAID"

LEGACY_METHOD_ALIASES: dict[str, str] = {
    "our": METHOD_VIGALT,
    "asset24": METHOD_CAID,
    "AAAA": METHOD_VIGALT,
    "ASSETS24": METHOD_CAID,
}

DEFAULT_VIGALT_CLAIMS_DIR = "output/ViGALT_claim_lists"
DEFAULT_CAID_CLAIMS_DIR = "output/CAID_claim_lists"

# Legacy typo path; migration script may rename this directory.
LEGACY_CAID_CLAIMS_DIR = "output/ASSEST24_claim_lists"

LEGACY_CAID_EXCEL_COLUMN = "ASSEST24 GENERATED SHORT CONTEXT-Aware ALT TEXT"


def normalize_method_id(value: str | None) -> str | None:
    if value is None:
        return None
    return LEGACY_METHOD_ALIASES.get(value, value)


def resolve_method_key(mapping: dict, canonical: str) -> str:
    """Return a key present in *mapping* for *canonical*, trying legacy aliases."""
    if canonical in mapping:
        return canonical
    for legacy, target in LEGACY_METHOD_ALIASES.items():
        if target == canonical and legacy in mapping:
            return legacy
    raise KeyError(f"No entry for method {canonical!r} in mapping (keys={list(mapping.keys())!r})")
