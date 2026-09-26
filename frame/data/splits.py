"""Geographic split assignment and leakage checks (Phase 3).

The rule (docs/Requirements 142.txt Requirement 3 section 32-34, Requirement 9 section 24):
splits are decided per GEOGRAPHIC UNIT, never per patch. Random patches from one region
land in both train and test and inflate every metric ("spatial leakage").

Units
-----
* scene  -- one source acquisition/ROI (`PairRecord.scene_id`). HARD rule, always: a scene
            never contributes samples to more than one split.
* region -- a geographic area containing scenes (`PairRecord.region_id`, e.g. a NEON site).
            Default rule: a region never spans splits either (`require_region_disjoint`).
Both are namespaced by dataset, so 'ABBY' in one dataset never collides with 'ABBY' in another.

Assignment
----------
`assign_splits` orders the units by a seeded SHA-256 (so it is reproducible and independent of
input order), then allocates them to train/val/test by cumulative weight so the split sizes
follow the requested fractions as closely as whole units allow. A dataset variant whose role
allows only some splits (SEN2NEON and OpenSR-Test: test only) is forced into them, and the
fractions are renormalised over the splits it may use. Explicit assignments override the hash
(for example a named Indian region -> test).

Checks
------
`check_split_integrity` reports, as `QCIssue`s: scene leakage, region leakage, role violations
(a benchmark in train), and -- as a warning -- samples from different splits that sit in the same
0.1 degree (~11 km) cell or in adjacent cells, which catches leakage between DIFFERENT datasets covering the same ground.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections import defaultdict
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from frame.data.contract import PairRecord, Split
from frame.data.errors import RoleViolationError, SplitError, SplitLeakageError
from frame.data.issues import QCIssue, error, warning
from frame.data.roles import DATASET_PROFILES

LEVELS = ("scene", "region")
PROXIMITY_CELL_DEG = 0.1


def unit_key(record: PairRecord, level: str) -> str:
    """The dataset-namespaced split unit of ``record`` at ``level``."""
    if level == "scene":
        return f"{record.dataset}/{record.scene_id}"
    if level == "region":
        if record.region_id is None:
            raise SplitError(
                f"{record.sample_id} has no region_id, so it cannot be split by region; give it a region or split by scene."
            )
        return f"{record.dataset}/{record.region_id}"
    raise SplitError(f"level must be one of {LEVELS}, got {level!r}.")


def allowed_splits(record: PairRecord) -> FrozenSet[Split]:
    """Splits the record's dataset variant is documented to be usable in (all three if the dataset is unknown)."""
    profile = DATASET_PROFILES.get(record.dataset)
    if profile is None or record.variant not in profile.variants:
        return frozenset(Split)
    return profile.variants[record.variant].allowed_splits


def _hash_order(units: Iterable[str], seed: int) -> List[str]:
    return sorted(units, key=lambda u: hashlib.sha256(f"{seed}|{u}".encode()).hexdigest())


def _allocate(units: Sequence[str], weights: Mapping[str, int], fractions: Mapping[Split, float]) -> Dict[str, Split]:
    """Hash-ordered units -> splits by cumulative weight; every split with a positive fraction gets a unit if there are enough."""
    active = [s for s in (Split.TRAIN, Split.VAL, Split.TEST) if fractions.get(s, 0.0) > 0]
    total_fraction = sum(fractions[s] for s in active)
    total_weight = sum(weights[u] for u in units)
    edges, running = [], 0.0
    for s in active:
        running += fractions[s] / total_fraction
        edges.append(running * total_weight)

    result: Dict[str, Split] = {}
    cumulative = 0.0
    for unit in units:
        middle = cumulative + weights[unit] / 2.0
        index = next((i for i, edge in enumerate(edges) if middle < edge), len(active) - 1)
        result[unit] = active[index]
        cumulative += weights[unit]

    if len(units) >= len(active):  # never leave a requested split empty
        for s in active:
            if s not in result.values():
                donor = max((c for c in active if list(result.values()).count(c) > 1),
                            key=lambda c: list(result.values()).count(c), default=None)
                if donor is not None:
                    victim = next(u for u in reversed(units) if result[u] == donor)
                    result[victim] = s
    return result


def assign_splits(
    records: Sequence[PairRecord],
    fractions: Mapping[Split, float],
    *,
    seed: int,
    level: str = "region",
    explicit: Optional[Mapping[str, Split]] = None,
    weight_by_records: bool = True,
) -> Dict[str, Split]:
    """Assign every geographic unit to one split. Returns ``{unit_key: Split}``.

    ``explicit`` maps namespaced unit keys (``"dataset/unit"``) to a split and overrides the hash.
    """
    if any(v < 0 for v in fractions.values()) or not any(v > 0 for v in fractions.values()):
        raise SplitError(f"fractions must be non-negative and not all zero, got {dict(fractions)}.")
    explicit = {k: Split(v) for k, v in (explicit or {}).items()}

    members: Dict[str, List[PairRecord]] = defaultdict(list)
    for record in records:
        members[unit_key(record, level)].append(record)
    unknown = set(explicit) - set(members)
    if unknown:
        raise SplitError(f"explicit assignment names unknown unit(s): {sorted(unknown)}.")

    weights = {u: (len(m) if weight_by_records else 1) for u, m in members.items()}
    permitted: Dict[str, FrozenSet[Split]] = {}
    for unit, recs in members.items():
        allowed = frozenset.intersection(*(allowed_splits(r) for r in recs))
        if not allowed:
            raise RoleViolationError(f"Unit {unit} mixes dataset variants whose roles share no split.")
        permitted[unit] = allowed

    assignment: Dict[str, Split] = {}
    for unit, split in explicit.items():
        if split not in permitted[unit]:
            raise RoleViolationError(f"Explicit split {split.value!r} for {unit} is not allowed by its dataset role ({sorted(s.value for s in permitted[unit])}).")
        assignment[unit] = split

    groups: Dict[FrozenSet[Split], List[str]] = defaultdict(list)
    for unit in members:
        if unit not in assignment:
            groups[permitted[unit]].append(unit)
    for allowed, units in groups.items():
        usable = {s: f for s, f in fractions.items() if s in allowed and f > 0}
        if not usable:  # the role permits only splits the caller gave no share to: use them equally
            usable = {s: 1.0 for s in allowed}
        assignment.update(_allocate(_hash_order(units, seed), weights, usable))
    return assignment


def apply_splits(records: Sequence[PairRecord], assignment: Mapping[str, Split], *, level: str = "region") -> List[PairRecord]:
    """Return copies of ``records`` with ``split`` set from ``assignment`` (the dataset's own label is kept in ``source_split``)."""
    return [dataclasses.replace(r, split=assignment[unit_key(r, level)]) for r in records]


# ---------------------------------------------------------------------------------------------------------------
# integrity checks
# ---------------------------------------------------------------------------------------------------------------

def check_split_integrity(
    records: Sequence[PairRecord],
    *,
    require_region_disjoint: bool = True,
    check_roles: bool = True,
    check_spatial_proximity: bool = True,
) -> List[QCIssue]:
    """All leakage / role findings for ``records`` (an empty list means the split is clean)."""
    issues: List[QCIssue] = []

    def spans(level: str) -> Dict[str, Dict[Split, List[str]]]:
        found: Dict[str, Dict[Split, List[str]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            if level == "region" and r.region_id is None:
                continue
            found[unit_key(r, level)][r.split].append(r.sample_id)
        return found

    for unit, by_split in sorted(spans("scene").items()):
        if len(by_split) > 1:
            ids = tuple(i for ids in by_split.values() for i in ids)
            issues.append(error("scene_leakage", f"scene {unit} contributes samples to {sorted(s.value for s in by_split)}: "
                                "a scene must belong to exactly one split.", *ids))
    for unit, by_split in sorted(spans("region").items()):
        if len(by_split) > 1:
            ids = tuple(i for ids in by_split.values() for i in ids)
            make = error if require_region_disjoint else warning
            issues.append(make("region_leakage", f"region {unit} spans splits {sorted(s.value for s in by_split)}: "
                               "split whole regions, not scenes within a region.", *ids))

    if check_roles:
        for r in records:
            allowed = allowed_splits(r)
            if r.split not in allowed:
                profile = DATASET_PROFILES.get(r.dataset)
                role = profile.variants[r.variant].role.value if profile and r.variant in profile.variants else "?"
                issues.append(error("role_violation", f"{r.dataset}/{r.variant} ({role}) may only be used in "
                                    f"{sorted(s.value for s in allowed)}, but {r.sample_id} is in {r.split.value!r}.", r.sample_id))

    if check_spatial_proximity:
        cells: Dict[Tuple[int, int], Dict[Split, List[str]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            if r.lat is not None and r.lon is not None:
                cells[(math.floor(r.lat / PROXIMITY_CELL_DEG), math.floor(r.lon / PROXIMITY_CELL_DEG))][r.split].append(r.sample_id)
        # Compare each cell with itself and its neighbours (each adjacent pair once): two samples a kilometre apart
        # can sit on either side of a cell boundary, so a same-cell test alone would miss them.
        for cell in sorted(cells):
            for d_lat, d_lon in ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
                neighbour = (cell[0] + d_lat, cell[1] + d_lon)
                if neighbour not in cells:
                    continue
                merged: Dict[Split, List[str]] = defaultdict(list)
                for source in {cell, neighbour}:
                    for split, ids in cells[source].items():
                        merged[split].extend(ids)
                if len(merged) > 1:
                    ids = tuple(dict.fromkeys(i for ids in merged.values() for i in ids))
                    issues.append(warning("spatial_proximity_across_splits",
                                          f"samples from splits {sorted(s.value for s in merged)} lie within one {PROXIMITY_CELL_DEG} degree "
                                          f"cell of each other around {cell}: check that different datasets do not cover the same ground.", *ids))
    return issues


def assert_no_leakage(records: Sequence[PairRecord], **kwargs) -> None:
    """Raise `SplitLeakageError` / `RoleViolationError` if `check_split_integrity` finds any error."""
    errors = [i for i in check_split_integrity(records, **kwargs) if i.severity == "error"]
    if not errors:
        return
    message = "; ".join(i.message for i in errors[:3]) + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")
    if any(i.code == "role_violation" for i in errors) and not any(i.code.endswith("leakage") for i in errors):
        raise RoleViolationError(message)
    raise SplitLeakageError(message)
