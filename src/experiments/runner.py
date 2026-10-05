"""Policy-independent validation sweeps and one-shot final-test runs.

Every configuration is run through the same simulator implementation. Each
policy/parameter run gets a fresh temporal state, generator, scaler, reward
index, and learner; an observed pool trace is compared within a sweep so a
policy cannot silently see a different schedule or candidate union.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.data.state import PlayEvent, TemporalState
from src.evaluation.metrics import OfflineMetricsEvaluator
from src.experiments.config import ExperimentConfig
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


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Metrics, provenance and decision log from one phase/policy run."""

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
    recall_undefined_decisions: int
    warmup: WarmupResult
    runtime_seconds: float
    created_at_utc: str
    candidate_pool_digest: str
    scaler_digest: str

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
            "n_users": len({record.user_id for record in self.simulation.decisions}),
            "n_decisions": self.simulation.decision_points,
            "skipped_decisions": self.simulation.skipped_decisions,
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


def _decision_to_dict(record: DecisionRecord) -> dict[str, Any]:
    return {
        "phase": record.phase,
        "user_id": record.user_id,
        "decision_time": record.decision_time.isoformat(),
        "selected_track": record.selected_track,
        "selected_artist": record.selected_artist,
        "selected_score": record.selected_score,
        "reward": record.reward,
        "reward_available_time": record.reward_available_time.isoformat() if record.reward_available_time else None,
        "candidate_count": record.candidate_count,
        "top_k_track_keys": [candidate.track_key for candidate in record.top_k_candidates],
        "top_k_artist_keys": [candidate.track.artist_key for candidate in record.top_k_candidates],
        "top_k_scores": list(record.top_k_scores),
        "top_k_effective_size": record.top_k_effective_size,
        "feedback_applied_before": record.feedback_applied_before,
    }


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
                (user_id, timestamp.isoformat(), tuple(c.track_key for c in pools.candidates)),
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
    and shared across every parameter run. Metrics are ranked only when the
    caller supplies/accepts the configured criterion; this class never picks
    a winning hyperparameter automatically.
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

    @property
    def eligible_users(self) -> frozenset[str]:
        """Identical eligibility set used by every policy and parameter run."""
        return self._eligible_users

    def _compute_eligibility(
        self,
    ) -> tuple[frozenset[str], dict[str, dict[str, int]], str]:
        """Count configured training units and phase-usable decision points."""
        if self.config.n_train_unit is None:
            raise ValueError(
                "n_train_unit must be explicitly set to 'events' or 'decision_points' "
                "before eligibility is computed"
            )
        protocol = self.config.temporal
        warm_start, warm_end = protocol.warmup_bounds
        val_start, val_end = protocol.validation_bounds
        test_start, test_end = protocol.test_bounds
        counts: dict[str, dict[str, int]] = {}
        dataset_digest = hashlib.sha256()

        for event in self.events:
            payload = json.dumps(
                (event.user_id, event.timestamp.isoformat(), event.track_id, event.artist_id),
                separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            dataset_digest.update(len(payload).to_bytes(8, "big"))
            dataset_digest.update(payload)
            if self.config.n_train_unit == "events" and warm_start <= event.timestamp <= warm_end:
                row = counts.setdefault(event.user_id, {"train": 0, "validation_usable": 0, "test_usable": 0})
                row["train"] += 1
        for session in iter_session_starts(self.events, protocol.session_gap):
            row = counts.setdefault(session.user_id, {"train": 0, "validation_usable": 0, "test_usable": 0})
            if self.config.n_train_unit == "decision_points" and warm_start <= session.decision_time <= warm_end:
                row["train"] += 1
            if val_start <= session.decision_time <= val_end and session.decision_time + protocol.reward_horizon <= val_end:
                row["validation_usable"] += 1
            if test_start <= session.decision_time <= test_end and session.decision_time + protocol.reward_horizon <= test_end:
                row["test_usable"] += 1

        eligible = frozenset(
            user_id for user_id, row in counts.items()
            if row["train"] >= self.config.n_min
            and row["validation_usable"] >= self.config.n_min
            and row["test_usable"] >= self.config.n_min
        )
        return eligible, counts, dataset_digest.hexdigest()

    def run_validation_sweep(self) -> tuple[ExperimentRun, ...]:
        """Run all configured validation combinations, without auto-selection."""
        if self.config.mode != "validation_sweep":
            raise ValueError("run_validation_sweep requires mode='validation_sweep'")
        if not self.eligible_users:
            raise ValueError("no users satisfy the configured N_min eligibility rule")
        runs: list[ExperimentRun] = []
        pool_traces: dict[int, tuple[str, str]] = {}
        for depth in self.config.candidate_pool_depths:
            for policy_name, params in self._validation_policy_grid():
                run = self._run_one("validation", policy_name, params, depth)
                signatures = (run.candidate_pool_digest, run.scaler_digest)
                previous = pool_traces.setdefault(depth, signatures)
                if previous != signatures:
                    raise RuntimeError(
                        "fairness violation: policy/parameter changed candidate pools or warm-up scaler"
                    )
                runs.append(run)
        return tuple(runs)

    def run_final_test(self) -> tuple[ExperimentRun, ...]:
        """Run one manually selected parameterization per policy on test only."""
        if self.config.mode != "final_test":
            raise ValueError("run_final_test requires mode='final_test'")
        if self._final_test_completed:
            raise RuntimeError("final test has already run for this ExperimentRunner")
        if not self.eligible_users:
            raise ValueError("no users satisfy the configured N_min eligibility rule")
        if len(self.config.candidate_pool_depths) != 1:
            raise ValueError("final_test requires exactly one selected candidate_pool_depth")
        missing = set(self.config.policies) - set(self.config.final_parameters)
        extra = set(self.config.final_parameters) - set(self.config.policies)
        if missing or extra:
            raise ValueError(f"final_parameters must match policies; missing={sorted(missing)}, extra={sorted(extra)}")
        depth = self.config.candidate_pool_depths[0]
        runs: list[ExperimentRun] = []
        expected_signatures = None
        for policy_name in self.config.policies:
            params = dict(self.config.final_parameters[policy_name])
            run = self._run_one("test", policy_name, params, depth)
            signatures = (run.candidate_pool_digest, run.scaler_digest)
            if expected_signatures is None:
                expected_signatures = signatures
            elif expected_signatures != signatures:
                raise RuntimeError("fairness violation: final-test candidate pools or scaler differ across policies")
            runs.append(run)
        self._final_test_completed = True
        return tuple(runs)

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
        self, phase: str, policy_name: str, params: Mapping[str, Any], depth: int
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
        warmup = simulator.prepare_warmup(warm_start, warm_end)
        scaler_digest = hashlib.sha256(
            json.dumps(
                (simulator.scaler.mean_, simulator.scaler.scale_),
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
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
            recall_undefined_decisions=evaluated.recall_excluded_decisions,
            warmup=warmup,
            runtime_seconds=perf_counter() - started,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            candidate_pool_digest=recording_generator.digest,
            scaler_digest=scaler_digest,
        )

    def rank_validation_results(
        self,
        runs: Sequence[ExperimentRun],
        *,
        metric: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> tuple[ExperimentRun, ...]:
        """Order runs by an explicit metric, placing undefined values last.

        This is a presentation operation only and never selects parameters or
        launches a test run. A criterion is mandatory either here or in config.
        """
        metric = metric or self.config.selection_metric
        direction = direction or self.config.selection_direction
        if not metric:
            raise ValueError("specify a selection metric before ranking validation results")
        if any(run.evaluation_phase != "validation" for run in runs):
            raise ValueError("only validation results may be ranked for parameter review")
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be maximize or minimize")
        sign = -1.0 if direction == "maximize" else 1.0
        return tuple(sorted(runs, key=lambda run: (
            run.metrics.get(metric) is None,
            sign * run.metrics[metric] if run.metrics.get(metric) is not None else 0.0,
            run.policy,
            json.dumps(dict(run.policy_parameters), sort_keys=True),
        )))

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
                stream.write(json.dumps(run.to_json_dict(self.config), sort_keys=True, ensure_ascii=False) + "\n")
        return csv_path, jsonl_path
