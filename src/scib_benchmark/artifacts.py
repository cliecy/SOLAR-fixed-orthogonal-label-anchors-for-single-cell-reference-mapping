from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutputTransaction:
    final: Path
    staging: Path | None
    reused: bool


def _archive_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.archived-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.archived-{stamp}-{suffix}")
        suffix += 1
    return candidate


def begin_output(final: Path, force: bool = False) -> OutputTransaction:
    status = final / "status.json"
    if final.exists():
        if status.is_file():
            try:
                if json.loads(status.read_text(encoding="utf-8")).get("state") == "completed" and not force:
                    return OutputTransaction(final, None, True)
            except json.JSONDecodeError:
                pass
        if not force:
            raise FileExistsError(
                f"Refusing to overwrite existing non-reusable output: {final}"
            )
        final.replace(_archive_path(final))
    staging = final.with_name(final.name + ".partial")
    if staging.exists():
        staging.replace(_archive_path(staging))
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    return OutputTransaction(final, staging, False)


def commit_output(transaction: OutputTransaction) -> Path:
    if transaction.reused:
        return transaction.final
    assert transaction.staging is not None
    if transaction.final.exists():
        raise FileExistsError(f"Final output appeared during execution: {transaction.final}")
    transaction.staging.replace(transaction.final)
    return transaction.final


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)

