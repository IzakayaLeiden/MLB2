from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class SkippedGame:
    game_id: int | None
    official_date: str | None
    reason: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    code: str
    message: str
    game_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    dataset: str
    row_count: int
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(issue.severity in {"critical", "high"} for issue in self.issues)

    def extend(self, issues: Iterable[QualityIssue]) -> None:
        self.issues.extend(issues)

    def to_dict(self) -> dict[str, Any]:
        counts = {severity: 0 for severity in ("critical", "high", "medium", "low")}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return {
            "dataset": self.dataset,
            "passed": self.passed,
            "row_count": self.row_count,
            "issue_counts": counts,
            "metrics": self.metrics,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class DataQualityError(RuntimeError):
    """차단 수준의 데이터 품질 문제가 발견됐을 때 발생합니다."""

