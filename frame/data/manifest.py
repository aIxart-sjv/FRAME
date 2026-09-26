"""Manifests: the reproducible index of a paired dataset (Phase 3).

A manifest is a JSON Lines file:

    line 1    {"_header": {"kind": "frame-paired-manifest", "manifest_version": 1, "n_records": N, ...}}
    line 2..  one `PairRecord` per line

Why a manifest and not a scan: the real datasets are 100+ GB and live outside the repository;
listing, splitting and checking them must not need their pixels. Every path in a record is
relative to its dataset directory (frame.data.config), so a manifest is portable and can be
committed without embedding anyone's machine paths.

Reproducibility: records are written sorted by ``sample_id`` with sorted keys and no timestamp,
so the same records always produce byte-identical files and the same `manifest_digest`; the
digest can be stored with a training run to prove which data it used.

Reading is strict by default (any bad line raises `ManifestError`); ``strict=False`` collects the
bad lines as issues instead so `frame.data.qc` can report all of them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from frame.data.contract import PairRecord
from frame.data.errors import ContractError, ManifestError
from frame.data.issues import QCIssue, error

MANIFEST_KIND = "frame-paired-manifest"
MANIFEST_VERSION = 1

PathLike = Union[str, Path]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest_digest(records: Iterable[PairRecord]) -> str:
    """SHA-256 over the canonical, sorted record lines (independent of file layout)."""
    digest = hashlib.sha256()
    for line in sorted(canonical_json(r.to_dict()) for r in records):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestContents:
    header: Dict[str, Any]
    records: List[PairRecord]
    issues: List[QCIssue]          # parse problems (only populated by strict=False reads)


def write_manifest(path: PathLike, records: Iterable[PairRecord], *, header: Optional[Dict[str, Any]] = None) -> str:
    """Write ``records`` as a manifest; returns its `manifest_digest`."""
    records = sorted(records, key=lambda r: r.sample_id)
    summary: Dict[str, Any] = {}
    for record in records:
        entry = summary.setdefault(record.dataset, {"variants": set(), "revisions": set()})
        entry["variants"].add(record.variant)
        if record.dataset_revision:
            entry["revisions"].add(record.dataset_revision)
    head = {
        "kind": MANIFEST_KIND, "manifest_version": MANIFEST_VERSION, "n_records": len(records),
        "datasets": {k: {"variants": sorted(v["variants"]), "revisions": sorted(v["revisions"])} for k, v in sorted(summary.items())},
        "digest": manifest_digest(records),
    }
    for key, value in (header or {}).items():
        if key in head:
            raise ManifestError(f"Header key {key!r} is reserved.")
        head[key] = value
    lines = [canonical_json({"_header": head})] + [canonical_json(r.to_dict()) for r in records]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return head["digest"]


def read_manifest(path: PathLike, *, strict: bool = True) -> ManifestContents:
    """Read a manifest. With ``strict=False`` a bad record line becomes an issue instead of an exception."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ManifestError(f"Manifest {path} is empty.")
    try:
        first = json.loads(lines[0])
        header = first["_header"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise ManifestError(f"{path}: line 1 must be the manifest header ({{\"_header\": ...}}).") from None
    if header.get("kind") != MANIFEST_KIND:
        raise ManifestError(f"{path}: not a {MANIFEST_KIND} (kind={header.get('kind')!r}).")
    if header.get("manifest_version") != MANIFEST_VERSION:
        raise ManifestError(f"{path}: unsupported manifest_version {header.get('manifest_version')!r} (this code reads {MANIFEST_VERSION}).")

    records: List[PairRecord] = []
    issues: List[QCIssue] = []
    for number, text in enumerate(lines[1:], start=2):
        if not text.strip():
            continue
        try:
            records.append(PairRecord.from_dict(json.loads(text)))
        except json.JSONDecodeError as exc:
            problem = error("invalid_json", f"line {number}: not valid JSON ({exc.msg}).", line=number)
        except ContractError as exc:
            problem = error(exc.code, f"line {number}: {exc}", line=number)
        else:
            continue
        if strict:
            raise ManifestError(problem.message)
        issues.append(problem)

    expected = header.get("n_records")
    if isinstance(expected, int) and expected != len(records) + len(issues):
        problem = error("record_count_mismatch",
                        f"header declares {expected} record(s) but the file holds {len(records) + len(issues)} (truncated or edited?).")
        if strict:
            raise ManifestError(problem.message)
        issues.append(problem)
    return ManifestContents(header=header, records=records, issues=issues)
