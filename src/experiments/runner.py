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
from pathlib import Path
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
            "created_at_utc": self.created_at_utc,
            "configuration_hash": self.configuration_hash,
            "evaluation_phase": self.evaluation_phase,
            "policy": self.policy,
            "policy_parameters": json.dumps(dict(self.policy_parameters), sort_keys=True),
            "alpha": self.policy_parameters.get("alpha"),
            "epsilon": self.policy_parameters.get("epsilon"),
            "dataset": self.dataset_id,
            "dataset_content_hash": self.dataset_content_hash,
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
            "selection_metric": config.selection_metric,
            "selection_direction": config.selection_direction,
            "eligible_user_hash": hashlib.sha256(
                "\n".join(sorted(self.eligible_users)).encode("utf-8")
            ).hexdigest(),
            "scaler_digest": self.scaler_digest,
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

    def run_validation_sweep(self) -> tuple[ExperimentRun, ...]:
        """Run the configured grid while retaining only compact run summaries.

        Detailed per-decision JSON is spooled to disk as each run completes;
        the returned records keep scalar counters and metrics but not the
        candidate/context-heavy decision tuple. This bounds retained sweep
        memory to one active simulation plus compact summaries.
        """
        if self.config.mode != "validation_sweep":
            raise ValueError("run_validation_sweep requires mode='validation_sweep'")
        if not self.eligible_users:
            raise ValueError("no users satisfy the configured N_min eligibility rule")
        runs: list[ExperimentRun] = []
        pool_traces: dict[int, tuple[str, str]] = {}
        spool_root = Path(self.config.results_directory)
        spool_root.mkdir(parents=True, exist_ok=True)
        spool = tempfile.TemporaryDirectory(
            prefix="music-discovery-validation-", dir=spool_root
        )
        try:
            for depth in self.config.candidate_pool_depths:
                for policy_name, params in self._validation_policy_grid():
                    full_run = self._run_one(
                        "validation", policy_name, params, depth,
                        warmup_artifact=self._warmup_artifacts.get(depth),
                    )
                    signatures = (full_run.candidate_pool_digest, full_run.scaler_digest)
                    previous = pool_traces.setdefault(depth, signatures)
                    if previous != signatures:
                        raise RuntimeError(
                            "fairness violation: policy/parameter changed candidate pools or warm-up scaler"
                        )
                    detail_path = Path(spool.name) / f"{full_run.run_id}.jsonl"
                    with detail_path.open("w", encoding="utf-8") as stream:
                        _write_run_json_line(stream, full_run, self.config)
                    compact_simulation = replace(
                        full_run.simulation,
                        decisions=(),
                        eligible_catalog=frozenset(),
                        eligible_artists=frozenset(),
                    )
                    runs.append(replace(
                        full_run,
                        simulation=compact_simulation,
                        details_path=str(detail_path),
                        _details_owner=spool,
                    ))
                    del full_run, compact_simulation
        except Exception:
            spool.cleanup()
            raise
        self._validation_spools.append(spool)
        return tuple(runs)

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
        recording_generator.start_recording()
        if phase == "validation":
            start, end = self.config.temporal.validation_bounds
        else:
            start, end = self.config.temporal.test_bounds
        simulation = simulator.run_phase(phase, start, end)
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
        identity = json.dumps(
            {
                "configuration_hash": self.config.configuration_hash,
                "dataset_content_hash": self.dataset_content_hash,
                "phase": phase,
                "policy": policy_name,
                "parameters": parameters,
                "depth": depth,
            }, sort_keys=True, separators=(",", ":")
        )
        run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return ExperimentRun(
            run_id=run_id,
            configuration_hash=self.config.configuration_hash,
            evaluation_phase=phase,
            policy=policy_name,
            policy_parameters=parameters,
            dataset_id=self.config.dataset_id,
            dataset_content_hash=self.dataset_content_hash,
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
