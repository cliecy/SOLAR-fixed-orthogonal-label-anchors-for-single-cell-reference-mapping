from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO


def checksum(path: Path, algorithm: str = "sha256", chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        _update_digest(handle, digest, chunk_size)
    return digest.hexdigest()


def _update_digest(handle: BinaryIO, digest: object, chunk_size: int) -> None:
    while chunk := handle.read(chunk_size):
        digest.update(chunk)  # type: ignore[attr-defined]


def duplicate_names(values: object) -> list[str]:
    import pandas as pd

    index = pd.Index(values)
    return sorted(set(index[index.duplicated(keep=False)].astype(str)))


def verify_file(path: Path, expected_size: int | None, expected_md5: str | None) -> dict[str, object]:
    if not path.is_file():
        return {"present": False, "size_matches": False, "md5_matches": False, "sha256": None}
    size = path.stat().st_size
    md5 = checksum(path, "md5") if expected_md5 else None
    return {"present": True, "size_bytes": size, "size_matches": expected_size is None or size == expected_size, "md5": md5, "md5_matches": expected_md5 is None or md5 == expected_md5, "sha256": checksum(path)}
