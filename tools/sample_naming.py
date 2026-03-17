from __future__ import annotations

def is_legacy_or_current_sample_name(filename: str) -> bool:
    return filename.endswith(".json") or filename.endswith(".npy")