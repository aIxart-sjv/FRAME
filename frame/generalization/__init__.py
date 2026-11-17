"""Domain-shift / geographic-generalisation protocol (Step H, Requirements 142 Requirement 12).

    frame.generalization.protocol   -- DomainShiftProtocol, indian_holdout_protocol
    frame.generalization.runner     -- run_domain_shift_experiment, DomainShiftRecord
    frame.generalization.report     -- write_record, render_readme
    frame.generalization.errors     -- GeneralizationError and its subclasses

An "Indian AOI alone is not evidence of Indian generalisation" (Requirements 142's own words):
this package always reports the protocol (training/test domain, geographic separation, land
cover, HR-reference status) alongside any measured value, never a bare number. There is
currently no Indian HR reference (docs/TRACEABILITY.md Sec 9), so nothing here computes or
implies reference-based accuracy for an Indian scene -- only self-consistency, index
self-consistency and TTA stability, all clearly labelled as such.
"""

from frame.generalization.errors import GeneralizationError, NoIndianSceneFoundError, UnsupportedProtocolStageError
from frame.generalization.protocol import (
    NO_REFERENCE_METRICS,
    TRAINING_DOMAIN_NOTE,
    DomainShiftProtocol,
    indian_holdout_protocol,
)
from frame.generalization.report import render_readme, write_record
from frame.generalization.runner import DomainShiftRecord, run_domain_shift_experiment

__all__ = [
    "GeneralizationError",
    "NoIndianSceneFoundError",
    "UnsupportedProtocolStageError",
    "DomainShiftProtocol",
    "indian_holdout_protocol",
    "NO_REFERENCE_METRICS",
    "TRAINING_DOMAIN_NOTE",
    "DomainShiftRecord",
    "run_domain_shift_experiment",
    "render_readme",
    "write_record",
]
