"""Policy-independent validation sweeps and one-shot final-test runs.

Every configuration is run through the same simulator implementation. Each
policy/parameter run gets a fresh temporal state, generator, scaler, reward
index, and learner; an observed pool trace is compared within a sweep so a
policy cannot silently see a different schedule or candidate union.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
import csv
from itertools import groupby
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.data.state import PlayEvent, TemporalState
from src.evaluation.metrics import OfflineMetricsEvaluator
from src.experiments.config import (
    DEFAULT_ALPHAS,
    DEFAULT_EPSILONS,
    PRIMARY_POOL_DEPTH,
    ExperimentConfig,
)
from src.recommender.policies import (
    EpsilonGreedyPolicy,
    GlobalPopularityPolicy,
    LinUCBPolicy,
    PersonalizedGreedyPolicy,
    RandomPolicy,
)
from src.simulation.candidate_generator import CandidateGenerator, CandidatePools
from src.simulation.context import ContextBuilder
from src.simulation.linucb import LinUCB
from src.simulation.reward import RewardCalculator
from src.simulation.scaler import FeatureScaler
from src.simulation.simulator import (
    DecisionRecord,
    SimulationResult,
    SequentialOfflineBanditSimulator,
    SessionStart,
    WarmupResult,
    iter_session_starts,
)
from types import MappingProxyType


_CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_CODE_FILES = (
    "src/data/identity.py",
    "src/data/state.py",
    "src/evaluation/metrics.py",
    "src/experiments/config.py",
    "src/experiments/dataset.py",
    "src/experiments/runner.py",
    "src/recommender/policies.py",
    "src/simulation/candidate_generator.py",
    "src/simulation/context.py",
    "src/simulation/linucb.py",
    "src/simulation/reward.py",
    "src/simulation/scaler.py",
    "src/simulation/simulator.py",
)


class CheckpointError(RuntimeError):
    """A persistent validation checkpoint is incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Metrics and provenance for one run; validation details may be disk-backed."""

    run_id: str
    configuration_hash: str
    evaluation_phase: str
    policy: str
    policy_parameters: Mapping[str, Any]
    dataset_id: str
    dataset_content_hash: str
    git_commit: Optional[str]
    experiment_code_digest: str
    random_seed: int
    candidate_pool_depth: int
    k_values: tuple[int, ...]
    eligible_users: frozenset[str]
    eligibility_counts: Mapping[str, Mapping[str, int]]
    simulation: SimulationResult
    metrics: Mapping[str, Optional[float]]
    users_evaluated: int
    evaluable_decisions: int
    empty_pool_decisions_excluded: int
    recall_undefined_decisions: int
    warmup: WarmupResult
    runtime_seconds: float
    warmup_runtime_seconds: float
    simulation_runtime_seconds: float
    created_at_utc: str
    candidate_pool_digest: str
    scaler_digest: str
    details_path: Optional[str] = None
    _details_owner: Any = field(default=None, repr=False, compare=False)

    def summary_row(self, config: ExperimentConfig) -> dict[str, Any]:
        """Flatten the per-run result into a CSV-friendly record."""
        temporal = config.temporal
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "completion_status": "completed",
            "created_at_utc": self.created_at_utc,
            "configuration_hash": self.configuration_hash,
            "evaluation_phase": self.evaluation_phase,
            "policy": self.policy,
            "policy_parameters": json.dumps(dict(self.policy_parameters), sort_keys=True),
            "alpha": self.policy_parameters.get("alpha"),
            "epsilon": self.policy_parameters.get("epsilon"),
            "dataset": self.dataset_id,
            "dataset_content_hash": self.dataset_content_hash,
            "git_commit": self.git_commit,
            "experiment_code_digest": self.experiment_code_digest,
            "dataset_path": config.dataset_path,
            "seed": config.seed,
            "random_seed": self.random_seed,
            "warmup_start": temporal.warmup_start.isoformat(),
            "warmup_end": temporal.warmup_end.isoformat(),
            "validation_start": temporal.validation_start.isoformat(),
            "validation_end": temporal.validation_end.isoformat(),
            "test_start": temporal.test_start.isoformat(),
            "test_end": temporal.test_end.isoformat(),
            "reward_horizon_days": temporal.reward_horizon_days,
            "session_gap_minutes": temporal.session_gap_minutes,
            "N_min": config.n_min,
            "N_train_unit": config.n_train_unit,
            "candidate_pool_depth": self.candidate_pool_depth,
            "K_values": ",".join(map(str, self.k_values)),
            "n_users": self.users_evaluated,
            "n_decisions": self.evaluable_decisions,
            "n_decision_points_total": self.simulation.decision_points,
            "skipped_decisions": self.simulation.skipped_decisions,
            "empty_pool_decisions_excluded": self.empty_pool_decisions_excluded,
            "feedback_scheduled": self.simulation.feedback_scheduled,
            "feedback_applied": self.simulation.feedback_applied,
            "feedback_pending_at_end": self.simulation.feedback_pending_at_end,
            "recall_undefined_decisions": self.recall_undefined_decisions,
            "runtime_seconds": self.runtime_seconds,
            "warmup_runtime_seconds": self.warmup_runtime_seconds,
            "simulation_runtime_seconds": self.simulation_runtime_seconds,
            "selection_metric": config.selection_metric,
            "selection_direction": config.selection_direction,
            "eligible_user_hash": hashlib.sha256(
                "\n".join(sorted(self.eligible_users)).encode("utf-8")
            ).hexdigest(),
            "scaler_digest": self.scaler_digest,
            "candidate_pool_digest": self.candidate_pool_digest,
            "warmup_decision_points": self.warmup.decision_points,
            "warmup_context_rows": self.warmup.context_rows,
            "warmup_state_events_processed": self.warmup.state_events_processed,
        }
        row.update({name: value for name, value in self.metrics.items()})
        return row

    def to_json_dict(self, config: ExperimentConfig) -> dict[str, Any]:
        """Return complete provenance and per-decision information as JSON data."""
        if self.details_path is not None:
            if not Path(self.details_path).is_file():
                raise RuntimeError("spooled validation details were closed or removed")
            with Path(self.details_path).open("r", encoding="utf-8") as stream:
                return json.load(stream)
        decisions = [_decision_to_dict(record) for record in self.simulation.decisions]
        return {
            "summary": self.summary_row(config),
            "eligible_users": sorted(self.eligible_users),
            "eligibility_counts": {
                user: dict(counts) for user, counts in self.eligibility_counts.items()
            },
            "candidate_pool_digest": self.candidate_pool_digest,
            "decisions": decisions,
        }


@dataclass(frozen=True, slots=True)
class ValidationSelection:
    """Parameters selected only from validation results at the primary N=100.

    Ties are resolved by higher Average Reward, then ascending parameter
    value. This object can build the common-N final-test configuration.
    """

    configuration_hash: str
    policy_parameters: Mapping[str, Mapping[str, Any]]
    selected_run_ids: Mapping[str, str]
    primary_pool_depth: int = PRIMARY_POOL_DEPTH
    criterion: str = "ndcg@10"

    def __post_init__(self) -> None:
        if self.primary_pool_depth != PRIMARY_POOL_DEPTH or self.criterion != "ndcg@10":
            raise ValueError("validation selection must use N=100 and ndcg@10")
        frozen_parameters = {
            name: MappingProxyType(dict(parameters))
            for name, parameters in self.policy_parameters.items()
        }
        object.__setattr__(self, "policy_parameters", MappingProxyType(frozen_parameters))
        object.__setattr__(self, "selected_run_ids", MappingProxyType(dict(self.selected_run_ids)))


@dataclass(frozen=True, slots=True)
class _WarmupArtifact:
    """In-memory, depth-specific scaler parameters and warm-up provenance."""

    configuration_hash: str
    dataset_content_hash: str
    candidate_pool_depth: int
    warmup_start: datetime
    warmup_end: datetime
    session_gap: timedelta
    result: WarmupResult
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    scaler_digest: str


def _current_git_commit(project_root: Path) -> Optional[str]:
    """Return the checked-out commit when this workspace is a Git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _experiment_code_digest(project_root: Path) -> str:
    """Hash every source module that can affect validation results."""
    digest = hashlib.sha256()
    for relative_path in _CHECKPOINT_CODE_FILES:
        path = project_root / relative_path
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"cannot fingerprint experiment source {relative_path!r}"
            ) from exc
        name = relative_path.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, writer) -> None:
    """Write one file in the target directory and atomically install it."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def _decision_to_dict(record: DecisionRecord) -> dict[str, Any]:
    return {
        "phase": record.phase,
        "user_id": record.user_id,
        "decision_time": record.decision_time.isoformat(),
        "selected_item_key": record.selected_item_key,
        "selected_artist": record.selected_artist,
        "selected_score": record.selected_score,
        "reward": record.reward,
        "reward_available_time": record.reward_available_time.isoformat() if record.reward_available_time else None,
        "candidate_count": record.candidate_count,
        "top_k_item_keys": [candidate.item_key for candidate in record.top_k_candidates],
        "top_k_artist_keys": [candidate.track.artist_key for candidate in record.top_k_candidates],
        "top_k_scores": list(record.top_k_scores),
        "top_k_effective_size": record.top_k_effective_size,
        "feedback_applied_before": record.feedback_applied_before,
    }


def _write_run_json_line(stream, run: ExperimentRun, config: ExperimentConfig) -> None:
    """Stream one complete run record without materializing all decisions."""
    stream.write("{\"summary\":")
    json.dump(run.summary_row(config), stream, sort_keys=True, ensure_ascii=False)
    stream.write(",\"eligible_users\":")
    json.dump(sorted(run.eligible_users), stream, ensure_ascii=False)
    stream.write(",\"eligibility_counts\":")
    json.dump(
        {user: dict(counts) for user, counts in run.eligibility_counts.items()},
        stream, sort_keys=True, ensure_ascii=False,
    )
    stream.write(",\"candidate_pool_digest\":")
    json.dump(run.candidate_pool_digest, stream)
    stream.write(',"decisions":[')
    for index, record in enumerate(run.simulation.decisions):
        if index:
            stream.write(",")
        json.dump(_decision_to_dict(record), stream, sort_keys=True, ensure_ascii=False)
    stream.write("]}\n")


class _RecordingGenerator:
    """Transparent generator adapter that records exact candidate unions."""

    def __init__(self, delegate: CandidateGenerator) -> None:
        self._delegate = delegate
        self.recording = False
        self._digest = hashlib.sha256()
        self.recorded_decisions = 0

    def generate(self, user_id: str, timestamp: datetime, pool_depth: int = 50) -> CandidatePools:
        pools = self._delegate.generate(user_id, timestamp, pool_depth)
        if self.recording:
            payload = json.dumps(
                (user_id, timestamp.isoformat(), tuple(c.item_key for c in pools.candidates)),
                separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            self._digest.update(len(payload).to_bytes(8, "big"))
            self._digest.update(payload)
            self.recorded_decisions += 1
        return pools

    def start_recording(self) -> None:
        self._digest = hashlib.sha256()
        self.recorded_decisions = 0
        self.recording = True

    @property
    def digest(self) -> str:
        return self._digest.hexdigest()


class ExperimentRunner:
    """Run explicit validation sweeps or final-test configurations.

    Eligibility and the temporal split are computed once from the event log
    and shared across every parameter run. Sensitivity runs use configured N
    values; selection is restricted to validation results at the fixed N=100.
    """

    def __init__(self, events: Sequence[PlayEvent], config: ExperimentConfig) -> None:
        if not isinstance(events, Sequence):
            raise TypeError("events must be a repeatable chronological Sequence[PlayEvent]")
        self.events = events
        self.config = config
        self._project_root = Path(__file__).resolve().parents[2]
        self.git_commit = _current_git_commit(self._project_root)
        self.experiment_code_digest = _experiment_code_digest(self._project_root)
        (
            self._eligible_users,
            self.eligibility_counts,
            self.dataset_content_hash,
        ) = self._compute_eligibility()
        self._reward_calculator = RewardCalculator(
            self.events, horizon=self.config.temporal.reward_horizon
        )
        self._metrics_evaluator = OfflineMetricsEvaluator(
            horizon=self.config.temporal.reward_horizon,
            reward_calculator=self._reward_calculator,
        )
        self._final_test_completed = False
        # Validation reuses only immutable warm-up artifacts, one per N. Each
        # run still owns fresh state, candidate index, scaler, and policy.
        self._warmup_artifacts: dict[int, _WarmupArtifact] = {}
        self.checkpoint_directory: Optional[Path] = None
        self.resumed_run_count = 0
        self.recovery_warmup_seconds: dict[int, float] = {}
        # Detailed validation decisions are spooled per run. Keeping the
        # TemporaryDirectory alive on the runner preserves write_results()
        # while allowing each heavy SimulationResult to be released at once.
        self._validation_spools: list[tempfile.TemporaryDirectory] = []

    @property
    def eligible_users(self) -> frozenset[str]:
        """Identical eligibility set used by every policy and parameter run."""
        return self._eligible_users

    def _compute_eligibility(
        self,
    ) -> tuple[frozenset[str], dict[str, dict[str, int]], str]:
        """Count configured training units and phase-usable decision points."""
        protocol = self.config.temporal
        warm_start, warm_end = protocol.warmup_bounds
        val_start, val_end = protocol.validation_bounds
        test_start, test_end = protocol.test_bounds
        counts: dict[str, dict[str, int]] = {}
        dataset_digest = hashlib.sha256()
        dataset_digest.update(b"identity-schema:typed-item-artist-framed-v2\0")

        for event in self.events:
            payload = json.dumps(
                (event.user_id, event.timestamp.isoformat(), event.item_key, event.artist_key),
                separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            dataset_digest.update(len(payload).to_bytes(8, "big"))
            dataset_digest.update(payload)
            if self.config.n_train_unit == "events" and warm_start <= event.timestamp <= warm_end:
                row = counts.setdefault(event.user_id, {"train": 0, "validation_usable": 0, "test_usable": 0})
                row["train"] += 1
        eligibility_state = TemporalState(self.events)
        eligibility_generator = CandidateGenerator(eligibility_state)
        sessions = iter_session_starts(self.events, protocol.session_gap)
        for timestamp, group_iter in groupby(sessions, key=lambda row: row.decision_time):
            group = tuple(group_iter)
            eligibility_state.advance_to(timestamp)
            for session in group:
                row = counts.setdefault(session.user_id, {"train": 0, "validation_usable": 0, "test_usable": 0})
                phase_counter = None
                phase_end = None
                if val_start <= timestamp <= val_end:
                    phase_counter, phase_end = "validation_usable", val_end
                elif test_start <= timestamp <= test_end:
                    phase_counter, phase_end = "test_usable", test_end
                if (
                    phase_counter is not None
                    and phase_end is not None
                    and timestamp + protocol.reward_horizon <= phase_end
                ):
                    # Eligibility is shared across configured N values. A
                    # point must have an action in every compared candidate
                    # pool; empty pools never count as usable.
                    has_candidates_at_every_depth = all(
                        eligibility_generator.generate(
                            session.user_id, timestamp, depth
                        ).candidates
                        for depth in self.config.candidate_pool_depths
                    )
                    if has_candidates_at_every_depth:
                        row[phase_counter] += 1
            # Keep all simultaneous session starts on the same strict-past
            # state, then reveal their boundary events for later timestamps.
            eligibility_state.advance_to(timestamp + timedelta(microseconds=1))

        eligible = frozenset(
            user_id for user_id, row in counts.items()
            if row["train"] >= self.config.n_min
            and row["validation_usable"] >= self.config.n_min
            and row["test_usable"] >= self.config.n_min
        )
        return eligible, counts, dataset_digest.hexdigest()

    def run_validation_sweep(
        self, checkpoint_directory: str | Path | None = None
    ) -> tuple[ExperimentRun, ...]:
        """Run or resume the validation grid from persistent per-run checkpoints.

        Each completed configuration is atomically written as a detailed JSONL
        record followed by a completion marker containing its checksum and
        provenance. A fresh runner can verify and reuse those records while
        rebuilding one independent warm-up artifact for each N.
        """
        if self.config.mode != "validation_sweep":
            raise ValueError("run_validation_sweep requires mode='validation_sweep'")
        if not self.eligible_users:
            raise ValueError("no users satisfy the configured N_min eligibility rule")
        checkpoint_root = self._resolve_checkpoint_directory(checkpoint_directory)
        self.checkpoint_directory = checkpoint_root
        planned: list[tuple[int, str, dict[str, Any], str]] = []
        for depth in self.config.candidate_pool_depths:
            for policy_name, params in self._validation_policy_grid():
                normalized_params = self._normalized_policy_parameters(policy_name, params)
                run_id = self._run_identifier(
                    "validation", policy_name, normalized_params, depth
                )
                planned.append((depth, policy_name, normalized_params, run_id))
        expected_run_ids = [row[3] for row in planned]
        if len(set(expected_run_ids)) != len(expected_run_ids):
            raise RuntimeError("validation grid generated duplicate run identifiers")

        completed = self._load_validation_checkpoints(
            checkpoint_root, planned
        )
        runs: list[ExperimentRun] = []
        pool_traces: dict[int, tuple[str, str]] = {}
        for depth in self.config.candidate_pool_depths:
            depth_plan = [row for row in planned if row[0] == depth]
            depth_has_checkpoints = any(row[3] in completed for row in depth_plan)
            if depth_has_checkpoints and depth not in self._warmup_artifacts:
                recovery_started = perf_counter()
                artifact = self._build_warmup_artifact(depth)
                self._warmup_artifacts[depth] = artifact
                self.recovery_warmup_seconds[depth] = perf_counter() - recovery_started
                for _, _, _, run_id in depth_plan:
                    cached = completed.get(run_id)
                    if cached is not None:
                        if cached["identity"]["scaler_digest"] != artifact.scaler_digest:
                            raise CheckpointError(
                                f"checkpoint {run_id} has a scaler digest incompatible with rebuilt warm-up"
                            )
                        warmup_summary = cached["summary"]
                        warmup_values = {
                            "warmup_decision_points": artifact.result.decision_points,
                            "warmup_context_rows": artifact.result.context_rows,
                            "warmup_state_events_processed": artifact.result.state_events_processed,
                        }
                        if any(
                            warmup_summary.get(name) != expected_value
                            for name, expected_value in warmup_values.items()
                        ):
                            raise CheckpointError(
                                f"checkpoint {run_id} warm-up counts differ from rebuilt artifact"
                            )

            for _, policy_name, params, run_id in depth_plan:
                cached = completed.get(run_id)
                if cached is not None:
                    run = self._restore_checkpoint_run(
                        cached, checkpoint_root / f"{run_id}.jsonl"
                    )
                    self.resumed_run_count += 1
                else:
                    run_started = perf_counter()
                    full_run = self._run_one(
                        "validation", policy_name, params, depth,
                        warmup_artifact=self._warmup_artifacts.get(depth),
                    )
                    full_run = replace(
                        full_run, runtime_seconds=perf_counter() - run_started
                    )
                    self._write_validation_checkpoint(checkpoint_root, full_run)
                    run = self._compact_checkpoint_run(
                        full_run, checkpoint_root / f"{run_id}.jsonl"
                    )
                    del full_run

                signatures = (run.candidate_pool_digest, run.scaler_digest)
                previous = pool_traces.setdefault(depth, signatures)
                if previous != signatures:
                    raise RuntimeError(
                        "fairness violation: policy/parameter changed candidate pools or warm-up scaler"
                    )
                runs.append(run)

        observed_ids = [run.run_id for run in runs]
        if len(observed_ids) != len(set(observed_ids)):
            raise CheckpointError("validation result aggregation contains duplicate run IDs")
        if set(observed_ids) != set(expected_run_ids) or len(observed_ids) != len(expected_run_ids):
            raise CheckpointError("validation result aggregation is missing planned configurations")
        return tuple(runs)

    def _resolve_checkpoint_directory(
        self, checkpoint_directory: str | Path | None
    ) -> Path:
        """Choose a persistent, ignored checkpoint directory for this sweep."""
        if checkpoint_directory is None:
            root = Path(self.config.results_directory) / "checkpoints" / (
                f"{self.config.configuration_hash[:12]}-{self.dataset_content_hash[:12]}"
            )
        else:
            root = Path(checkpoint_directory)
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        try:
            root.relative_to(self._project_root)
        except ValueError:
            pass
        else:
            ignored = subprocess.run(
                ["git", "check-ignore", "--quiet", str(root)],
                cwd=self._project_root,
                check=False,
                capture_output=True,
            )
            if ignored.returncode != 0:
                raise ValueError(
                    "checkpoint directory inside the repository must be ignored by Git"
                )
        return root

    def _normalized_policy_parameters(
        self, policy_name: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        normalized = dict(params)
        if policy_name in {"random", "epsilon_greedy"}:
            normalized["random_seed"] = int(
                normalized.get("random_seed", self.config.random_seed)
            )
        return normalized

    def _run_identifier(
        self,
        phase: str,
        policy_name: str,
        params: Mapping[str, Any],
        depth: int,
    ) -> str:
        identity = json.dumps(
            {
                "configuration_hash": self.config.configuration_hash,
                "dataset_content_hash": self.dataset_content_hash,
                "phase": phase,
                "policy": policy_name,
                "parameters": dict(params),
                "depth": depth,
            }, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    def _build_warmup_artifact(self, depth: int) -> _WarmupArtifact:
        """Rebuild one N-specific scaler and fresh state after a resumed run."""
        state = TemporalState(self.events)
        generator = CandidateGenerator(state)
        simulator = SequentialOfflineBanditSimulator(
            events=self.events,
            eligible_users=self.eligible_users,
            state=state,
            candidate_generator=generator,
            context_builder=ContextBuilder(state),
            scaler=FeatureScaler(),
            reward_calculator=self._reward_calculator,
            policy=self._policy("random", {}),
            candidate_pool_depth=depth,
            top_k=self.config.top_k,
            session_gap=self.config.temporal.session_gap,
            reward_horizon=self.config.temporal.reward_horizon,
        )
        warm_start, warm_end = self.config.temporal.warmup_bounds
        try:
            result = simulator.prepare_warmup(warm_start, warm_end)
            mean = tuple(simulator.scaler.mean_)
            scale = tuple(simulator.scaler.scale_)
        finally:
            generator.close()
        return _WarmupArtifact(
            configuration_hash=self.config.configuration_hash,
            dataset_content_hash=self.dataset_content_hash,
            candidate_pool_depth=depth,
            warmup_start=warm_start,
            warmup_end=warm_end,
            session_gap=self.config.temporal.session_gap,
            result=result,
            scaler_mean=mean,
            scaler_scale=scale,
            scaler_digest=self._scaler_digest(mean, scale),
        )

    def _checkpoint_identity(self, run: ExperimentRun) -> dict[str, Any]:
        eligible_hash = hashlib.sha256(
            "\n".join(sorted(run.eligible_users)).encode("utf-8")
        ).hexdigest()
        return {
            "run_id": run.run_id,
            "configuration_hash": run.configuration_hash,
            "dataset_content_hash": run.dataset_content_hash,
            "experiment_code_digest": run.experiment_code_digest,
            "git_commit": run.git_commit,
            "evaluation_phase": run.evaluation_phase,
            "policy": run.policy,
            "policy_parameters": dict(run.policy_parameters),
            "candidate_pool_depth": run.candidate_pool_depth,
            "seed": self.config.seed,
            "random_seed": run.random_seed,
            "eligible_users_hash": eligible_hash,
            "eligible_user_count": len(run.eligible_users),
            "scaler_digest": run.scaler_digest,
            "candidate_pool_digest": run.candidate_pool_digest,
        }

    def _write_validation_checkpoint(
        self, checkpoint_root: Path, run: ExperimentRun
    ) -> None:
        detail_path = checkpoint_root / f"{run.run_id}.jsonl"
        marker_path = checkpoint_root / f"{run.run_id}.checkpoint.json"
        if detail_path.exists() or marker_path.exists():
            raise CheckpointError(
                f"refusing to overwrite an existing checkpoint for run {run.run_id}"
            )
        _atomic_write(
            detail_path,
            lambda stream: _write_run_json_line(stream, run, self.config),
        )
        details_hash = _sha256_file(detail_path)
        marker = {
            "checkpoint_format_version": _CHECKPOINT_FORMAT_VERSION,
            "completion_status": "completed",
            "identity": self._checkpoint_identity(run),
            "summary": run.summary_row(self.config),
            "metrics": dict(run.metrics),
            "eligible_users": sorted(run.eligible_users),
            "eligibility_counts": {
                user: dict(counts)
                for user, counts in run.eligibility_counts.items()
            },
            "timings": {
                "total_seconds": run.runtime_seconds,
                "warmup_or_restore_seconds": run.warmup_runtime_seconds,
                "simulation_seconds": run.simulation_runtime_seconds,
            },
            "result_filename": detail_path.name,
            "result_sha256": details_hash,
        }

        def write_marker(stream) -> None:
            json.dump(marker, stream, sort_keys=True, ensure_ascii=False)
            stream.write("\n")

        _atomic_write(marker_path, write_marker)

    def _load_validation_checkpoints(
        self,
        checkpoint_root: Path,
        planned: Sequence[tuple[int, str, dict[str, Any], str]],
    ) -> dict[str, dict[str, Any]]:
        expected = {row[3]: row for row in planned}
        stale_temporary = tuple(checkpoint_root.glob("*.tmp")) + tuple(
            checkpoint_root.glob(".*.tmp")
        )
        if stale_temporary:
            raise CheckpointError(
                "incomplete atomic checkpoint write found: "
                + ", ".join(path.name for path in stale_temporary)
            )
        marker_paths = tuple(sorted(checkpoint_root.glob("*.checkpoint.json")))
        marker_ids = {path.name.removesuffix(".checkpoint.json") for path in marker_paths}
        orphan_details = tuple(
            path for path in checkpoint_root.glob("*.jsonl")
            if path.stem not in marker_ids
        )
        if orphan_details:
            raise CheckpointError(
                "checkpoint detail file has no completion marker: "
                + ", ".join(path.name for path in orphan_details)
            )

        completed: dict[str, dict[str, Any]] = {}
        eligible_hash = hashlib.sha256(
            "\n".join(sorted(self.eligible_users)).encode("utf-8")
        ).hexdigest()
        for marker_path in marker_paths:
            run_id = marker_path.name.removesuffix(".checkpoint.json")
            if run_id not in expected:
                raise CheckpointError(
                    f"checkpoint {marker_path.name} does not belong to the configured grid"
                )
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointError(f"corrupt checkpoint marker {marker_path}") from exc
            if marker.get("checkpoint_format_version") != _CHECKPOINT_FORMAT_VERSION:
                raise CheckpointError(f"unsupported checkpoint format in {marker_path}")
            if marker.get("completion_status") != "completed":
                raise CheckpointError(f"checkpoint is not marked complete: {marker_path}")
            identity = marker.get("identity")
            if not isinstance(identity, dict):
                raise CheckpointError(f"checkpoint identity is missing: {marker_path}")
            depth, policy_name, params, expected_id = expected[run_id]
            expected_identity = {
                "run_id": expected_id,
                "configuration_hash": self.config.configuration_hash,
                "dataset_content_hash": self.dataset_content_hash,
                "experiment_code_digest": self.experiment_code_digest,
                "git_commit": self.git_commit,
                "evaluation_phase": "validation",
                "policy": policy_name,
                "policy_parameters": params,
                "candidate_pool_depth": depth,
                "seed": self.config.seed,
                "random_seed": int(params.get("random_seed", self.config.random_seed)),
                "eligible_users_hash": eligible_hash,
                "eligible_user_count": len(self.eligible_users),
            }
            for key, value in expected_identity.items():
                if identity.get(key) != value:
                    raise CheckpointError(
                        f"checkpoint {run_id} is incompatible: {key} differs"
                    )
            detail_name = f"{run_id}.jsonl"
            if marker.get("result_filename") != detail_name:
                raise CheckpointError(f"checkpoint result filename mismatch: {marker_path}")
            detail_path = checkpoint_root / detail_name
            if not detail_path.is_file():
                raise CheckpointError(f"checkpoint result is missing: {detail_path}")
            if _sha256_file(detail_path) != marker.get("result_sha256"):
                raise CheckpointError(f"checkpoint result checksum mismatch: {detail_path}")
            summary = marker.get("summary")
            if not isinstance(summary, dict):
                raise CheckpointError(f"checkpoint summary is missing: {marker_path}")
            summary_expected = {
                "run_id": run_id,
                "completion_status": "completed",
                "configuration_hash": self.config.configuration_hash,
                "dataset_content_hash": self.dataset_content_hash,
                "git_commit": self.git_commit,
                "experiment_code_digest": self.experiment_code_digest,
                "evaluation_phase": "validation",
                "policy": policy_name,
                "policy_parameters": json.dumps(params, sort_keys=True),
                "candidate_pool_depth": depth,
                "seed": self.config.seed,
                "random_seed": int(params.get("random_seed", self.config.random_seed)),
                "eligible_user_hash": eligible_hash,
            }
            for key, value in summary_expected.items():
                if summary.get(key) != value:
                    raise CheckpointError(
                        f"checkpoint result {run_id} is incompatible: summary {key} differs"
                    )
            if identity.get("scaler_digest") != summary.get("scaler_digest"):
                raise CheckpointError(f"checkpoint scaler digest mismatch: {run_id}")
            if identity.get("candidate_pool_digest") != summary.get("candidate_pool_digest"):
                raise CheckpointError(f"checkpoint candidate digest mismatch: {run_id}")
            eligible_users = marker.get("eligible_users")
            if not isinstance(eligible_users, list) or hashlib.sha256(
                "\n".join(sorted(eligible_users)).encode("utf-8")
            ).hexdigest() != eligible_hash:
                raise CheckpointError(f"checkpoint eligible-user list mismatch: {run_id}")
            result_counts = marker.get("eligibility_counts")
            if not isinstance(result_counts, dict) or set(result_counts) != set(self.eligibility_counts):
                raise CheckpointError(f"checkpoint eligibility counts mismatch: {run_id}")
            normalized_counts = {
                user: {name: int(value) for name, value in counts.items()}
                for user, counts in result_counts.items()
            }
            if normalized_counts != self.eligibility_counts:
                raise CheckpointError(f"checkpoint eligibility values mismatch: {run_id}")
            if not isinstance(marker.get("metrics"), dict):
                raise CheckpointError(f"checkpoint metrics are missing: {run_id}")
            for key, value in marker["metrics"].items():
                if summary.get(key) != value:
                    raise CheckpointError(f"checkpoint metric mismatch for {key}: {run_id}")
            timings = marker.get("timings")
            if not isinstance(timings, dict) or any(
                not isinstance(timings.get(name), (int, float))
                or not math.isfinite(timings[name])
                or timings[name] < 0
                for name in (
                    "total_seconds", "warmup_or_restore_seconds", "simulation_seconds"
                )
            ):
                raise CheckpointError(f"checkpoint timings are invalid: {run_id}")
            if summary.get("n_decision_points_total") != summary.get("n_decisions", 0) + summary.get("empty_pool_decisions_excluded", 0):
                raise CheckpointError(f"checkpoint decision counts are inconsistent: {run_id}")
            if not isinstance(identity.get("candidate_pool_digest"), str) or len(identity["candidate_pool_digest"]) != 64:
                raise CheckpointError(f"checkpoint candidate digest is invalid: {run_id}")
            if marker["timings"]["total_seconds"] != summary.get("runtime_seconds"):
                raise CheckpointError(f"checkpoint total runtime mismatch: {run_id}")
            if marker["timings"]["warmup_or_restore_seconds"] != summary.get("warmup_runtime_seconds"):
                raise CheckpointError(f"checkpoint warm-up runtime mismatch: {run_id}")
            if marker["timings"]["simulation_seconds"] != summary.get("simulation_runtime_seconds"):
                raise CheckpointError(f"checkpoint simulation runtime mismatch: {run_id}")
            completed[run_id] = marker
        return completed

    def _restore_checkpoint_run(
        self,
        marker: Mapping[str, Any],
        detail_path: Path,
    ) -> ExperimentRun:
        summary = marker["summary"]
        identity = marker["identity"]
        phase = str(summary["evaluation_phase"])
        simulation = SimulationResult(
            phase=phase,
            decisions=(),
            decision_points=int(summary["n_decision_points_total"]),
            skipped_decisions=int(summary["skipped_decisions"]),
            feedback_scheduled=int(summary["feedback_scheduled"]),
            feedback_applied=int(summary["feedback_applied"]),
            feedback_pending_at_end=int(summary["feedback_pending_at_end"]),
        )
        warmup = WarmupResult(
            decision_points=int(summary["warmup_decision_points"]),
            context_rows=int(summary["warmup_context_rows"]),
            state_events_processed=int(summary["warmup_state_events_processed"]),
        )
        return ExperimentRun(
            run_id=str(summary["run_id"]),
            configuration_hash=str(summary["configuration_hash"]),
            evaluation_phase=phase,
            policy=str(summary["policy"]),
            policy_parameters=json.loads(summary["policy_parameters"]),
            dataset_id=str(summary["dataset"]),
            dataset_content_hash=str(summary["dataset_content_hash"]),
            git_commit=summary.get("git_commit"),
            experiment_code_digest=str(summary["experiment_code_digest"]),
            random_seed=int(summary["random_seed"]),
            candidate_pool_depth=int(summary["candidate_pool_depth"]),
            k_values=self.config.k_values,
            eligible_users=frozenset(marker["eligible_users"]),
            eligibility_counts={
                user: {name: int(value) for name, value in counts.items()}
                for user, counts in marker["eligibility_counts"].items()
            },
            simulation=simulation,
            metrics=dict(marker["metrics"]),
            users_evaluated=int(summary["n_users"]),
            evaluable_decisions=int(summary["n_decisions"]),
            empty_pool_decisions_excluded=int(summary["empty_pool_decisions_excluded"]),
            recall_undefined_decisions=int(summary["recall_undefined_decisions"]),
            warmup=warmup,
            runtime_seconds=float(marker["timings"]["total_seconds"]),
            warmup_runtime_seconds=float(marker["timings"]["warmup_or_restore_seconds"]),
            simulation_runtime_seconds=float(marker["timings"]["simulation_seconds"]),
            created_at_utc=str(summary["created_at_utc"]),
            candidate_pool_digest=str(identity["candidate_pool_digest"]),
            scaler_digest=str(identity["scaler_digest"]),
            details_path=str(detail_path),
        )

    @staticmethod
    def _compact_checkpoint_run(run: ExperimentRun, detail_path: Path) -> ExperimentRun:
        compact_simulation = replace(
            run.simulation,
            decisions=(),
            eligible_catalog=frozenset(),
            eligible_artists=frozenset(),
        )
        return replace(run, simulation=compact_simulation, details_path=str(detail_path))

    def close(self) -> None:
        """Release temporary per-run validation detail files owned by runner."""
        for spool in self._validation_spools:
            spool.cleanup()
        self._validation_spools.clear()

    def run_final_test(self) -> tuple[ExperimentRun, ...]:
        """Run one explicitly selected, common-N parameterization per policy."""
        if self.config.mode != "final_test":
            raise ValueError("run_final_test requires mode='final_test'")
        if self._final_test_completed:
            raise RuntimeError("final test has already run for this ExperimentRunner")
        if not self.eligible_users:
            raise ValueError("no users satisfy the configured N_min eligibility rule")
        if len(self.config.candidate_pool_depths) != 1:
            raise ValueError("final_test requires exactly one selected candidate_pool_depth")
        if self.config.candidate_pool_depths != (PRIMARY_POOL_DEPTH,):
            raise ValueError("final_test requires the common primary candidate_pool_depth N=100")
        missing = set(self.config.policies) - set(self.config.final_parameters)
        extra = set(self.config.final_parameters) - set(self.config.policies)
        if missing or extra:
            raise ValueError(f"final_parameters must match policies; missing={sorted(missing)}, extra={sorted(extra)}")
        depth = self.config.candidate_pool_depths[0]
        runs: list[ExperimentRun] = []
        expected_signatures = None
        for policy_name in self.config.policies:
            params = dict(self.config.final_parameters[policy_name])
            if policy_name == "linucb" and float(params.get("alpha", -1.0)) not in DEFAULT_ALPHAS:
                raise ValueError(f"final LinUCB alpha must come from {DEFAULT_ALPHAS}")
            if policy_name == "epsilon_greedy" and float(params.get("epsilon", -1.0)) not in DEFAULT_EPSILONS:
                raise ValueError(f"final epsilon must come from {DEFAULT_EPSILONS}")
            run = self._run_one("test", policy_name, params, depth)
            signatures = (run.candidate_pool_digest, run.scaler_digest)
            if expected_signatures is None:
                expected_signatures = signatures
            elif expected_signatures != signatures:
                raise RuntimeError("fairness violation: final-test candidate pools or scaler differ across policies")
            runs.append(run)
        self._final_test_completed = True
        return tuple(runs)

    def select_validation_parameters(
        self, runs: Sequence[ExperimentRun]
    ) -> ValidationSelection:
        """Select alpha and epsilon from validation at N=100 only.

        N=50 and N=200 runs are ignored for parameter selection and remain
        descriptive sensitivity analyses. The selector accepts validation
        rows only, so test metrics cannot influence this decision.
        """
        if self.config.mode != "validation_sweep":
            raise ValueError("parameter selection requires mode='validation_sweep'")
        if not runs:
            raise ValueError("validation results are required for parameter selection")
        if any(run.evaluation_phase != "validation" for run in runs):
            raise ValueError("selection accepts validation results only; test results are forbidden")
        if any(run.configuration_hash != self.config.configuration_hash for run in runs):
            raise ValueError("validation results do not match this ExperimentConfig")
        if any(run.dataset_content_hash != self.dataset_content_hash for run in runs):
            raise ValueError("validation results do not match this dataset")

        primary_runs = tuple(
            run for run in runs if run.candidate_pool_depth == PRIMARY_POOL_DEPTH
        )
        selected: dict[str, Mapping[str, Any]] = {}
        selected_ids: dict[str, str] = {}
        for policy in self.config.policies:
            candidates = tuple(run for run in primary_runs if run.policy == policy)
            if not candidates:
                raise ValueError(f"missing N=100 validation results for policy {policy!r}")
            if policy == "linucb":
                expected = {float(value) for value in self.config.alpha_values}
                observed = {
                    float(run.policy_parameters["alpha"])
                    for run in candidates
                    if "alpha" in run.policy_parameters
                }
                if observed != expected or len(candidates) != len(expected):
                    raise ValueError("N=100 LinUCB results must cover every configured alpha exactly once")
            elif policy == "epsilon_greedy":
                expected = {float(value) for value in self.config.epsilon_values}
                observed = {
                    float(run.policy_parameters["epsilon"])
                    for run in candidates
                    if "epsilon" in run.policy_parameters
                }
                if observed != expected or len(candidates) != len(expected):
                    raise ValueError("N=100 epsilon-greedy results must cover every configured epsilon exactly once")
            elif len(candidates) != 1:
                raise ValueError(f"non-tunable policy {policy!r} must have one N=100 validation result")

            winner = (
                self._best_validation_run(candidates)
                if policy in {"linucb", "epsilon_greedy"}
                else candidates[0]
            )
            selected[policy] = dict(winner.policy_parameters)
            selected_ids[policy] = winner.run_id

        return ValidationSelection(
            configuration_hash=self.config.configuration_hash,
            policy_parameters=selected,
            selected_run_ids=selected_ids,
        )

    @staticmethod
    def _best_validation_run(runs: Sequence[ExperimentRun]) -> ExperimentRun:
        """Order by NDCG@10, Average Reward, then lower parameter value."""
        scored = [run for run in runs if run.metrics.get("ndcg@10") is not None]
        if not scored:
            raise ValueError("NDCG@10 is undefined for every candidate configuration")

        def parameter_order(run: ExperimentRun) -> tuple[float, str]:
            parameter = run.policy_parameters.get("alpha", run.policy_parameters.get("epsilon"))
            return (float(parameter) if parameter is not None else -1.0, run.policy)

        def key(run: ExperimentRun) -> tuple[Any, ...]:
            reward = run.metrics.get("average_reward")
            return (
                -float(run.metrics["ndcg@10"]),
                reward is None,
                -float(reward) if reward is not None else 0.0,
                *parameter_order(run),
                run.run_id,
            )

        return min(scored, key=key)

    def make_final_test_config(
        self, selection: ValidationSelection
    ) -> ExperimentConfig:
        """Build the one-shot N=100 test config from this validation selection."""
        if self.config.mode != "validation_sweep":
            raise ValueError("a final-test config can be built only from a validation runner")
        if selection.configuration_hash != self.config.configuration_hash:
            raise ValueError("selection was produced from a different validation config")
        if set(selection.policy_parameters) != set(self.config.policies):
            raise ValueError("selection must provide one parameter mapping for every configured policy")
        return replace(
            self.config,
            mode="final_test",
            candidate_pool_depths=(PRIMARY_POOL_DEPTH,),
            final_parameters={
                name: dict(parameters)
                for name, parameters in selection.policy_parameters.items()
            },
            selection_metric="ndcg@10",
            selection_direction="maximize",
        )

    def _validation_policy_grid(self) -> Iterable[tuple[str, dict[str, Any]]]:
        for name in self.config.policies:
            if name == "linucb":
                for alpha in self.config.alpha_values:
                    yield name, {"alpha": alpha}
            elif name == "epsilon_greedy":
                for epsilon in self.config.epsilon_values:
                    yield name, {"epsilon": epsilon}
            else:
                yield name, {}

    def _policy(self, name: str, params: Mapping[str, Any]):
        allowed_parameters = {
            "random": {"random_seed"},
            "popularity": set(),
            "greedy": set(),
            "epsilon_greedy": {"epsilon", "random_seed"},
            "linucb": {"alpha"},
        }
        if name not in allowed_parameters:
            raise ValueError(f"unknown policy {name!r}")
        unknown = set(params) - allowed_parameters[name]
        if unknown:
            raise ValueError(f"unsupported {name} parameters: {sorted(unknown)}")
        if name == "random":
            return RandomPolicy(int(params.get("random_seed", self.config.random_seed)))
        if name == "popularity":
            return GlobalPopularityPolicy()
        if name == "greedy":
            return PersonalizedGreedyPolicy()
        if name == "epsilon_greedy":
            if "epsilon" not in params:
                raise ValueError("epsilon_greedy requires epsilon")
            return EpsilonGreedyPolicy(float(params["epsilon"]), int(params.get("random_seed", self.config.random_seed)))
        if "alpha" not in params:
            raise ValueError("linucb requires alpha")
        return LinUCBPolicy(LinUCB(alpha=float(params["alpha"])))

    def _run_one(
        self,
        phase: str,
        policy_name: str,
        params: Mapping[str, Any],
        depth: int,
        *,
        warmup_artifact: Optional[_WarmupArtifact] = None,
    ) -> ExperimentRun:
        started = perf_counter()
        state = TemporalState(self.events)
        delegate = CandidateGenerator(state)
        recording_generator = _RecordingGenerator(delegate)
        policy = self._policy(policy_name, params)
        simulator = SequentialOfflineBanditSimulator(
            events=self.events,
            eligible_users=self.eligible_users,
            state=state,
            candidate_generator=recording_generator,  # type: ignore[arg-type]
            context_builder=ContextBuilder(state),
            scaler=FeatureScaler(),
            reward_calculator=self._reward_calculator,
            policy=policy,
            candidate_pool_depth=depth,
            top_k=self.config.top_k,
            session_gap=self.config.temporal.session_gap,
            reward_horizon=self.config.temporal.reward_horizon,
        )
        warm_start, warm_end = self.config.temporal.warmup_bounds
        warmup_started = perf_counter()
        if warmup_artifact is None:
            warmup = simulator.prepare_warmup(warm_start, warm_end)
            scaler_mean = tuple(simulator.scaler.mean_)
            scaler_scale = tuple(simulator.scaler.scale_)
            scaler_digest = self._scaler_digest(scaler_mean, scaler_scale)
            if phase == "validation":
                self._warmup_artifacts[depth] = _WarmupArtifact(
                    configuration_hash=self.config.configuration_hash,
                    dataset_content_hash=self.dataset_content_hash,
                    candidate_pool_depth=depth,
                    warmup_start=warm_start,
                    warmup_end=warm_end,
                    session_gap=self.config.temporal.session_gap,
                    result=warmup,
                    scaler_mean=scaler_mean,
                    scaler_scale=scaler_scale,
                    scaler_digest=scaler_digest,
                )
        else:
            self._validate_warmup_artifact(warmup_artifact, depth, warm_start, warm_end)
            warmup = simulator.prepare_warmup_from_precomputed(
                warm_start,
                warm_end,
                scaler_mean=warmup_artifact.scaler_mean,
                scaler_scale=warmup_artifact.scaler_scale,
                expected=warmup_artifact.result,
            )
            scaler_digest = self._scaler_digest(
                simulator.scaler.mean_, simulator.scaler.scale_
            )
            if scaler_digest != warmup_artifact.scaler_digest:
                raise RuntimeError("imported warm-up scaler digest differs from its artifact")
        warmup_runtime = perf_counter() - warmup_started
        recording_generator.start_recording()
        if phase == "validation":
            start, end = self.config.temporal.validation_bounds
        else:
            start, end = self.config.temporal.test_bounds
        simulation_started = perf_counter()
        simulation = simulator.run_phase(phase, start, end)
        simulation_runtime = perf_counter() - simulation_started
        recording_generator.recording = False
        if recording_generator.recorded_decisions != simulation.decision_points:
            raise RuntimeError("candidate generator must run exactly once per decision point")
        evaluated = self._metrics_evaluator.evaluate(
            simulation, k_values=self.config.k_values
        )
        # Break TemporalState -> listener -> CandidateGenerator -> state so
        # the completed run's large temporal indexes are released promptly.
        delegate.close()
        metrics = {name: summary.value for name, summary in evaluated.metrics.items()}
        parameters = dict(params)
        if policy_name in {"random", "epsilon_greedy"}:
            parameters["random_seed"] = int(params.get("random_seed", self.config.random_seed))
        run_id = self._run_identifier(phase, policy_name, parameters, depth)
        return ExperimentRun(
            run_id=run_id,
            configuration_hash=self.config.configuration_hash,
            evaluation_phase=phase,
            policy=policy_name,
            policy_parameters=parameters,
            dataset_id=self.config.dataset_id,
            dataset_content_hash=self.dataset_content_hash,
            git_commit=self.git_commit,
            experiment_code_digest=self.experiment_code_digest,
            random_seed=int(parameters.get("random_seed", self.config.random_seed)),
            candidate_pool_depth=depth,
            k_values=self.config.k_values,
            eligible_users=self.eligible_users,
            eligibility_counts=self.eligibility_counts,
            simulation=simulation,
            metrics=metrics,
            users_evaluated=evaluated.users_evaluated,
            evaluable_decisions=evaluated.decisions_evaluated,
            empty_pool_decisions_excluded=sum(
                record.candidate_count == 0 for record in simulation.decisions
            ),
            recall_undefined_decisions=evaluated.recall_excluded_decisions,
            warmup=warmup,
            runtime_seconds=perf_counter() - started,
            warmup_runtime_seconds=warmup_runtime,
            simulation_runtime_seconds=simulation_runtime,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            candidate_pool_digest=recording_generator.digest,
            scaler_digest=scaler_digest,
        )

    @staticmethod
    def _scaler_digest(
        mean: Sequence[float], scale: Sequence[float]
    ) -> str:
        return hashlib.sha256(
            json.dumps((tuple(mean), tuple(scale)), separators=(",", ":"))
            .encode("ascii")
        ).hexdigest()

    def _validate_warmup_artifact(
        self,
        artifact: _WarmupArtifact,
        depth: int,
        warm_start: datetime,
        warm_end: datetime,
    ) -> None:
        """Reject in-memory artifacts from any different run provenance."""
        expected = (
            self.config.configuration_hash,
            self.dataset_content_hash,
            depth,
            warm_start,
            warm_end,
            self.config.temporal.session_gap,
        )
        observed = (
            artifact.configuration_hash,
            artifact.dataset_content_hash,
            artifact.candidate_pool_depth,
            artifact.warmup_start,
            artifact.warmup_end,
            artifact.session_gap,
        )
        if observed != expected:
            raise ValueError("warm-up artifact provenance does not match this configuration")
        if self._scaler_digest(artifact.scaler_mean, artifact.scaler_scale) != artifact.scaler_digest:
            raise ValueError("warm-up artifact scaler parameters do not match their digest")

    def rank_validation_results(
        self,
        runs: Sequence[ExperimentRun],
        *,
        candidate_pool_depth: int = PRIMARY_POOL_DEPTH,
    ) -> tuple[ExperimentRun, ...]:
        """Order validation rows by NDCG@10, then frozen tie-breaks.

        The default N=100 view is the primary selection table. Passing 50 or
        200 creates a descriptive sensitivity-analysis ranking only.
        """
        if any(run.evaluation_phase != "validation" for run in runs):
            raise ValueError("only validation results may be ranked for parameter review")
        if candidate_pool_depth not in {50, 100, 200}:
            raise ValueError("candidate_pool_depth must be 50, 100, or 200")
        rows = tuple(run for run in runs if run.candidate_pool_depth == candidate_pool_depth)
        return tuple(sorted(rows, key=self._best_validation_sort_key))

    @staticmethod
    def _best_validation_sort_key(run: ExperimentRun) -> tuple[Any, ...]:
        ndcg = run.metrics.get("ndcg@10")
        reward = run.metrics.get("average_reward")
        parameter = run.policy_parameters.get("alpha", run.policy_parameters.get("epsilon"))
        return (
            ndcg is None,
            -float(ndcg) if ndcg is not None else 0.0,
            reward is None,
            -float(reward) if reward is not None else 0.0,
            float(parameter) if parameter is not None else -1.0,
            run.policy,
            run.run_id,
        )

    def write_results(
        self, runs: Sequence[ExperimentRun], output_directory: str | Path | None = None
    ) -> tuple[Path, Path]:
        """Write flattened CSV and full JSONL, one self-contained run per line."""
        directory = Path(output_directory or self.config.results_directory)
        phase = runs[0].evaluation_phase if runs else self.config.mode
        directory = directory / phase / (
            f"{self.config.configuration_hash[:12]}-{self.dataset_content_hash[:8]}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "runs.csv"
        jsonl_path = directory / "runs.jsonl"
        rows = [run.summary_row(self.config) for run in runs]
        fieldnames = sorted({name for row in rows for name in row})
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with jsonl_path.open("w", encoding="utf-8") as stream:
            for run in runs:
                if run.details_path is not None:
                    if not Path(run.details_path).is_file():
                        raise RuntimeError(
                            "spooled validation details were closed or removed before writing results"
                        )
                    with Path(run.details_path).open("r", encoding="utf-8") as detail_stream:
                        shutil.copyfileobj(detail_stream, stream)
                else:
                    _write_run_json_line(stream, run, self.config)
        return csv_path, jsonl_path
