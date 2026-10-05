"""Typed, JSON-serializable experiment configuration.

Dates in configuration files are inclusive calendar dates. They are expanded
to the final microsecond of the given day so the existing simulator's
inclusive timestamp bounds preserve the frozen split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


DEFAULT_POLICIES = ("random", "popularity", "greedy", "epsilon_greedy", "linucb")
SUPPORTED_POOL_DEPTHS = (50, 100, 200)
SUPPORTED_K_VALUES = (5, 10, 20)
DEFAULT_ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
DEFAULT_EPSILONS = (0.05, 0.10, 0.20)


def _date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO calendar date: {value!r}") from exc


def _positive_unique(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class TemporalProtocol:
    """Frozen warm-up, validation, test and event-session timing bounds."""

    warmup_start: date = date(2005, 10, 30)
    warmup_end: date = date(2008, 12, 31)
    validation_start: date = date(2009, 1, 1)
    validation_end: date = date(2009, 2, 28)
    test_start: date = date(2009, 3, 1)
    test_end: date = date(2009, 5, 5)
    reward_horizon_days: int = 7
    session_gap_minutes: int = 10

    def __post_init__(self) -> None:
        normalized = {name: _date(getattr(self, name)) for name in (
            "warmup_start", "warmup_end", "validation_start",
            "validation_end", "test_start", "test_end",
        )}
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        if self.warmup_end < self.warmup_start:
            raise ValueError("warm-up end must not precede start")
        if self.validation_start <= self.warmup_end:
            raise ValueError("validation must begin after warm-up")
        if self.validation_end < self.validation_start:
            raise ValueError("validation end must not precede start")
        if self.test_start <= self.validation_end:
            raise ValueError("test must begin after validation")
        if self.test_end < self.test_start:
            raise ValueError("test end must not precede start")
        if self.reward_horizon_days <= 0 or self.session_gap_minutes <= 0:
            raise ValueError("reward horizon and session gap must be positive")

    @staticmethod
    def _start(day: date) -> datetime:
        return datetime.combine(day, time.min)

    @staticmethod
    def _end(day: date) -> datetime:
        return datetime.combine(day, time.max)

    @property
    def warmup_bounds(self) -> tuple[datetime, datetime]:
        return self._start(self.warmup_start), self._end(self.warmup_end)

    @property
    def validation_bounds(self) -> tuple[datetime, datetime]:
        return self._start(self.validation_start), self._end(self.validation_end)

    @property
    def test_bounds(self) -> tuple[datetime, datetime]:
        return self._start(self.test_start), self._end(self.test_end)

    @property
    def reward_horizon(self) -> timedelta:
        return timedelta(days=self.reward_horizon_days)

    @property
    def session_gap(self) -> timedelta:
        return timedelta(minutes=self.session_gap_minutes)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete run configuration; parameter selection is always explicit.

    ``mode`` is ``validation_sweep`` or ``final_test``. Validation values are
    ranked for inspection only; no best parameter is selected automatically.
    Final test requires a single explicit ``final_parameters`` entry per
    configured policy.
    """

    dataset_path: str = "Dataset/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
    dataset_id: str = "lastfm-dataset-1K"
    results_directory: str = "experiments/results"
    seed: int = 42
    random_seed: int = 42
    mode: str = "validation_sweep"
    temporal: TemporalProtocol = field(default_factory=TemporalProtocol)
    n_min: int = 10
    n_train_unit: str | None = "events"
    candidate_pool_depths: tuple[int, ...] = (50,)
    k_values: tuple[int, ...] = (5, 10, 20)
    policies: tuple[str, ...] = DEFAULT_POLICIES
    alpha_values: tuple[float, ...] = DEFAULT_ALPHAS
    epsilon_values: tuple[float, ...] = DEFAULT_EPSILONS
    final_parameters: Mapping[str, Mapping[str, float | int]] = field(default_factory=dict)
    selection_metric: str | None = None
    selection_direction: str = "maximize"

    def __post_init__(self) -> None:
        if self.mode not in {"validation_sweep", "final_test"}:
            raise ValueError("mode must be validation_sweep or final_test")
        if isinstance(self.n_min, bool) or not isinstance(self.n_min, int) or self.n_min <= 0:
            raise ValueError("n_min must be a positive integer")
        if self.n_train_unit not in {None, "events", "decision_points"}:
            raise ValueError("n_train_unit must be events, decision_points, or unset")
        if not self.dataset_path.strip() or not self.dataset_id.strip():
            raise ValueError("dataset_path and dataset_id must be nonblank")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if not self.candidate_pool_depths or any(
            depth not in SUPPORTED_POOL_DEPTHS for depth in self.candidate_pool_depths
        ):
            raise ValueError(f"candidate_pool_depths must use {SUPPORTED_POOL_DEPTHS}")
        if len(set(self.candidate_pool_depths)) != len(self.candidate_pool_depths):
            raise ValueError("candidate_pool_depths must not contain duplicates")
        _positive_unique(self.k_values, "k_values")
        if set(self.k_values) - set(SUPPORTED_K_VALUES):
            raise ValueError(f"k_values must use {SUPPORTED_K_VALUES}")
        allowed = set(DEFAULT_POLICIES)
        if not self.policies or len(set(self.policies)) != len(self.policies) or set(self.policies) - allowed:
            raise ValueError(f"policies must be unique names from {sorted(allowed)}")
        if not self.alpha_values or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            for value in self.alpha_values
        ):
            raise ValueError("alpha_values must be nonempty and nonnegative")
        if not self.epsilon_values or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1
            for value in self.epsilon_values
        ):
            raise ValueError("epsilon_values must be nonempty and in [0, 1]")
        if self.selection_direction not in {"maximize", "minimize"}:
            raise ValueError("selection_direction must be maximize or minimize")
        if self.mode == "final_test" and not self.final_parameters:
            raise ValueError("final_test requires explicitly selected final_parameters")
        if self.mode == "validation_sweep" and self.final_parameters:
            raise ValueError("final_parameters belongs only to final_test configuration")

    @property
    def top_k(self) -> int:
        return max(self.k_values)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe configuration data with ISO date strings."""
        data = asdict(self)
        data["temporal"] = {
            name: getattr(self.temporal, name).isoformat()
            for name in (
                "warmup_start", "warmup_end", "validation_start",
                "validation_end", "test_start", "test_end",
            )
        } | {
            "reward_horizon_days": self.temporal.reward_horizon_days,
            "session_gap_minutes": self.temporal.session_gap_minutes,
        }
        for key in ("candidate_pool_depths", "k_values", "policies", "alpha_values", "epsilon_values"):
            data[key] = list(data[key])
        return data

    @property
    def configuration_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        payload = dict(data)
        temporal_data = dict(payload.pop("temporal", {}))
        if temporal_data:
            for name in ("warmup_start", "warmup_end", "validation_start", "validation_end", "test_start", "test_end"):
                if name in temporal_data:
                    temporal_data[name] = _date(temporal_data[name])
            payload["temporal"] = TemporalProtocol(**temporal_data)
        for name in ("candidate_pool_depths", "k_values", "policies", "alpha_values", "epsilon_values"):
            if name in payload:
                payload[name] = tuple(payload[name])
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
