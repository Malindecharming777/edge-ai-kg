"""Shared ETL helpers (parsing, normalization, id minting)."""

def norm_id(prefix: str, value: str) -> str:
    return f"{prefix}:{value.strip().lower().replace(' ', '_')}"
