"""The one issue type shared by manifest reading, split checks and QC (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class QCIssue:
    """One finding. ``severity`` 'error' makes the affected samples invalid; 'warning' does not."""

    code: str
    severity: str
    message: str
    sample_ids: Tuple[str, ...] = ()
    line: Optional[int] = None          # manifest line number, for parse problems

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "message": self.message,
                "sample_ids": list(self.sample_ids), "line": self.line}


def error(code: str, message: str, *sample_ids: str, line: Optional[int] = None) -> QCIssue:
    return QCIssue(code, ERROR, message, tuple(sample_ids), line)


def warning(code: str, message: str, *sample_ids: str, line: Optional[int] = None) -> QCIssue:
    return QCIssue(code, WARNING, message, tuple(sample_ids), line)
