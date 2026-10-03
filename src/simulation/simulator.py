"""Sequential offline evaluation of the existing contextual bandit modules.

Policy observations advance through historical events strictly before each
decision. Future events are queried only through ``RewardCalculator`` and
never enter contexts or candidate indexes until the observable clock reaches
them. Session starts sharing an exact timestamp are processed atomically so
none of their boundary events leaks into another decision at that time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import heapq
from typing import Iterable, Iterator, Optional, Sequence

from src.data.state import PlayEvent, TemporalState
from src.simulation.candidate_generator import (
    Candidate,
    CandidateGenerator,
    CandidatePools,
    SUPPORTED_POOL_DEPTHS,
)
from src.simulation.context import ContextBuilder, ContextVector
from src.simulation.linucb import LinUCB
from src.simulation.reward import DEFAULT_REWARD_HORIZON, RewardCalculator
from src.simulation.scaler import FeatureScaler


DEFAULT_SESSION_GAP = timedelta(minutes=10)
WARMUP_START = datetime(2005, 10, 30)
WARMUP_END = datetime(2008, 12, 31, 23, 59, 59, 999999)
VALIDATION_START = datetime(2009, 1, 1)
VALIDATION_END = datetime(2009, 2, 28, 23, 59, 59, 999999)
TEST_START = datetime(2009, 3, 1)
TEST_END = datetime(2009, 5, 5, 23, 59, 59, 999999)


def _aware(timestamp: datetime) -> bool:
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _utc_naive(timestamp: datetime) -> datetime:
    """Normalize an aware instant to UTC while preserving naive timestamps."""
    if _aware(timestamp):
        return timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp


def _add_elapsed(timestamp: datetime, duration: timedelta) -> datetime:
    """Add elapsed time, including across daylight-saving transitions."""
    if _aware(timestamp):
        return (timestamp.astimezone(timezone.utc) + duration).astimezone(
            timestamp.tzinfo
        )
    return timestamp + duration


def _elapsed_between(earlier: datetime, later: datetime) -> timedelta:
    """Calculate elapsed duration rather than wall-clock duration."""
    return _utc_naive(later) - _utc_naive(earlier)


@dataclass(frozen=True, slots=True)
class SessionStart:
    """First event and decision time of one user's listening session."""

    user_id: str
    decision_time: datetime
    boundary_event: PlayEvent


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Immutable decision log row; empty pools have no selected action/reward."""

    phase: str
    user_id: str
    decision_time: datetime
    selected_candidate: Optional[Candidate]
    selected_artist: Optional[str]
    selected_track: Optional[str]
    selected_score: Optional[float]
    selected_context: Optional[ContextVector]
    selected_scaled_context: Optional[ContextVector]
    reward: Optional[float]
    reward_available_time: Optional[datetime]
    candidate_count: int
    top_k_candidates: tuple[Candidate, ...]
    top_k_scores: tuple[float, ...]
    top_k_effective_size: int
    linucb_mean: Optional[float]
    linucb_exploration: Optional[float]
    feedback_applied_before: int


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Per-phase decision logs and feedback counts, without aggregate metrics."""

    phase: str
    decisions: tuple[DecisionRecord, ...]
    decision_points: int
    skipped_decisions: int
    feedback_scheduled: int
    feedback_applied: int
    feedback_pending_at_end: int


@dataclass(frozen=True, slots=True)
class WarmupResult:
    """Warm-up work summary; no policy decisions or learner updates occur."""

    decision_points: int
    context_rows: int
    state_events_processed: int


@dataclass(frozen=True, slots=True)
class _PendingFeedback:
    availability_time: datetime
    decision_time: datetime
    context: ContextVector
    reward: float


def iter_session_starts(
    events: Iterable[PlayEvent],
    session_gap: timedelta = DEFAULT_SESSION_GAP,
) -> Iterator[SessionStart]:
    """Yield per-user session starts from globally chronological events.

    A new session begins only when the gap from that user's previous event is
    strictly greater than ``session_gap``. The input is streamed with O(U)
    auxiliary memory, where U is the number of users.
    """
    if not isinstance(session_gap, timedelta) or session_gap <= timedelta(0):
        raise ValueError("session_gap must be a positive timedelta")

    last_by_user: dict[str, datetime] = {}
    last_global_time: Optional[datetime] = None
    timezone_aware: Optional[bool] = None
    for event in events:
        if not isinstance(event, PlayEvent):
            raise TypeError("events must contain PlayEvent instances")
        if not isinstance(event.user_id, str) or not event.user_id.strip():
            raise ValueError("each event must have a nonblank user_id")
        if not isinstance(event.timestamp, datetime):
            raise ValueError("each event must have a datetime timestamp")
        event_is_aware = _aware(event.timestamp)
        if timezone_aware is None:
            timezone_aware = event_is_aware
        elif event_is_aware != timezone_aware:
            raise ValueError("event timestamps must use a consistent timezone convention")
        if last_global_time is not None and event.timestamp < last_global_time:
            raise ValueError("events must be ordered chronologically")

        previous = last_by_user.get(event.user_id)
        if previous is None or _elapsed_between(previous, event.timestamp) > session_gap:
            yield SessionStart(event.user_id, event.timestamp, event)
        last_by_user[event.user_id] = event.timestamp
        last_global_time = event.timestamp


def _group_by_decision_time(
    starts: Iterable[SessionStart],
) -> Iterator[tuple[datetime, tuple[SessionStart, ...]]]:
    """Group simultaneous decisions so every one sees the same strict past."""
    current_time: Optional[datetime] = None
    group: list[SessionStart] = []
    for session in starts:
        if current_time is None:
            current_time = session.decision_time
        if session.decision_time != current_time:
            yield current_time, tuple(group)
            current_time = session.decision_time
            group = []
        group.append(session)
    if current_time is not None:
        yield current_time, tuple(group)


class SequentialOfflineBanditSimulator:
    """Orchestrate warm-up, chronological validation/test and delayed updates.

    The caller provides the shared chronological event sequence, an already
    computed eligible-user set, and the existing state/policy components.
    ``prepare_warmup`` builds the state/candidate history and fits an unfitted
    scaler from raw candidate contexts drawn only from warm-up. If the scaler
    is already fitted externally, warm-up still advances the state but never
    refits it. Validation and test each start with a fresh LinUCB learner; the
    same temporal state and candidate index continue across both phases.
    """

    def __init__(
        self,
        events: Sequence[PlayEvent],
        eligible_users: Iterable[str],
        state: TemporalState,
        candidate_generator: CandidateGenerator,
        context_builder: ContextBuilder,
        scaler: FeatureScaler,
        learner: LinUCB,
        reward_calculator: RewardCalculator,
        *,
        candidate_pool_depth: int = 50,
        top_k: int = 10,
        session_gap: timedelta = DEFAULT_SESSION_GAP,
        reward_horizon: timedelta = DEFAULT_REWARD_HORIZON,
    ) -> None:
        if candidate_pool_depth not in SUPPORTED_POOL_DEPTHS:
            raise ValueError(
                f"candidate_pool_depth must be one of {sorted(SUPPORTED_POOL_DEPTHS)}"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(session_gap, timedelta) or session_gap <= timedelta(0):
            raise ValueError("session_gap must be a positive timedelta")
        if not isinstance(reward_horizon, timedelta) or reward_horizon <= timedelta(0):
            raise ValueError("reward_horizon must be a positive timedelta")

        # Repeatable input is required for warm-up, validation, and test scans.
        if not isinstance(events, Sequence):
            raise TypeError(
                "events must be a repeatable sequence (for example, a list or tuple)"
            )
        self._events = events
        self._eligible_users = frozenset(eligible_users)
        if any(not isinstance(user, str) or not user.strip() for user in self._eligible_users):
            raise ValueError("eligible_users must contain nonblank user IDs")
        self._state = state
        self._candidate_generator = candidate_generator
        self._context_builder = context_builder
        self._scaler = scaler
        self._learner = learner
        if learner.n_features != 15:
            raise ValueError("LinUCB n_features must match the 15-feature context")
        self._reward_calculator = reward_calculator
        self._candidate_pool_depth = candidate_pool_depth
        self._top_k = top_k
        self._session_gap = session_gap
        self._reward_horizon = reward_horizon

        self._pending: list[tuple[int, int, _PendingFeedback]] = []
        self._feedback_sequence = 0
        self._warmup_prepared = False
        self._last_phase: Optional[str] = None

    @property
    def state(self) -> TemporalState:
        """The continuing observable temporal state."""
        return self._state

    @property
    def learner(self) -> LinUCB:
        """Current phase learner (replaced with a fresh model per phase)."""
        return self._learner

    @property
    def scaler(self) -> FeatureScaler:
        """The single scaler instance shared across validation and test."""
        return self._scaler

    def _session_groups(self) -> Iterator[tuple[datetime, tuple[SessionStart, ...]]]:
        return _group_by_decision_time(iter_session_starts(self._events, self._session_gap))

    def _contexts_for_user(
        self, user_id: str, timestamp: datetime
    ) -> tuple[CandidatePools, tuple[ContextVector, ...]]:
        pools = self._candidate_generator.generate(
            user_id, timestamp, self._candidate_pool_depth
        )
        contexts = tuple(
            self._context_builder.build(user_id, candidate, timestamp)
            for candidate in pools.candidates
        )
        return pools, contexts

    @staticmethod
    def _after_inclusive_time(timestamp: datetime) -> datetime:
        return _add_elapsed(timestamp, timedelta(microseconds=1))

    def prepare_warmup(
        self,
        start: datetime = WARMUP_START,
        end: datetime = WARMUP_END,
    ) -> WarmupResult:
        """Advance history through warm-up and fit only from warm-up contexts.

        Warm-up samples are all generated candidates for eligible users at
        warm-up session starts. The scaler counts each candidate occurrence
        once. No reward is queried/scheduled and no LinUCB update is performed.
        """
        if self._warmup_prepared:
            raise RuntimeError("warm-up has already been prepared")
        if not isinstance(start, datetime) or not isinstance(end, datetime) or end < start:
            raise ValueError("warm-up bounds must be valid datetimes with end >= start")
        if self._state.current_time is not None and self._state.current_time > start:
            raise ValueError("TemporalState has already advanced beyond warm-up start")

        counts = {"decisions": 0, "contexts": 0}

        def warmup_contexts() -> Iterator[ContextVector]:
            for timestamp, sessions in self._session_groups():
                if timestamp < start:
                    continue
                if timestamp > end:
                    break
                self._state.advance_to(timestamp)
                for session in sessions:
                    if session.user_id not in self._eligible_users:
                        continue
                    counts["decisions"] += 1
                    pools, contexts = self._contexts_for_user(session.user_id, timestamp)
                    counts["contexts"] += len(contexts)
                    yield from contexts
                # Make all events at t observable only after simultaneous decisions.
                self._state.advance_to(self._after_inclusive_time(timestamp))
            if self._state.current_time is None or self._state.current_time <= end:
                self._state.advance_to(self._after_inclusive_time(end))

        before_updates = (self._learner.A, self._learner.b)
        if self._scaler.is_fitted:
            for _ in warmup_contexts():
                pass
        else:
            self._scaler.fit(warmup_contexts())
        if (self._learner.A, self._learner.b) != before_updates:
            raise RuntimeError("warm-up must not update LinUCB")
        self._warmup_prepared = True
        return WarmupResult(
            decision_points=counts["decisions"],
            context_rows=counts["contexts"],
            state_events_processed=self._state.processed_event_count,
        )

    def _process_pending(self, timestamp: datetime) -> int:
        current_us = self._time_key(timestamp)
        applied = 0
        while self._pending and self._pending[0][0] <= current_us:
            _, _, feedback = heapq.heappop(self._pending)
            self._learner.update(feedback.context, feedback.reward)
            applied += 1
        return applied

    @staticmethod
    def _time_key(timestamp: datetime) -> int:
        epoch = datetime(1970, 1, 1)
        delta = _utc_naive(timestamp) - epoch
        return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds

    def _schedule_feedback(
        self, decision_time: datetime, context: ContextVector, reward: float
    ) -> datetime:
        availability = _add_elapsed(decision_time, self._reward_horizon)
        feedback = _PendingFeedback(availability, decision_time, context, reward)
        heapq.heappush(
            self._pending,
            (self._time_key(availability), self._feedback_sequence, feedback),
        )
        self._feedback_sequence += 1
        return availability

    def run_phase(
        self,
        phase: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> SimulationResult:
        """Run one validation or test phase in chronological session order.

        Only sessions whose decision time is inside the phase and whose entire
        reward window satisfies ``t + horizon <= end`` are evaluated. Test and
        validation each reset LinUCB; observable history/indexes are retained.
        """
        if phase not in {"validation", "test"}:
            raise ValueError("phase must be 'validation' or 'test'")
        defaults = {
            "validation": (VALIDATION_START, VALIDATION_END),
            "test": (TEST_START, TEST_END),
        }
        default_start, default_end = defaults[phase]
        start = default_start if start is None else start
        end = default_end if end is None else end
        if not isinstance(start, datetime) or not isinstance(end, datetime) or end < start:
            raise ValueError("phase bounds must be valid datetimes with end >= start")
        if not self._warmup_prepared:
            raise RuntimeError("prepare_warmup must be called before a phase")
        if not self._scaler.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted on warm-up data")
        if self._pending:
            raise RuntimeError("pending feedback must be empty at phase start")
        if self._state.current_time is not None and start < self._state.current_time:
            raise ValueError("phase start precedes the current observable state")

        # The musical history persists; the learner is reset for each phase.
        self._learner = LinUCB(
            n_features=self._learner.n_features, alpha=self._learner.alpha
        )
        decisions: list[DecisionRecord] = []
        feedback_scheduled = 0
        feedback_applied = 0
        skipped = 0

        for timestamp, sessions in self._session_groups():
            if timestamp < start:
                continue
            if timestamp > end:
                break
            usable_sessions = tuple(
                session
                for session in sessions
                if session.user_id in self._eligible_users
                and _add_elapsed(timestamp, self._reward_horizon) <= end
            )
            if not usable_sessions:
                continue

            feedback_applied += self._process_pending(timestamp)
            self._state.advance_to(timestamp)

            for session in usable_sessions:
                pools, raw_contexts = self._contexts_for_user(
                    session.user_id, timestamp
                )
                if not pools.candidates:
                    decisions.append(
                        DecisionRecord(
                            phase=phase,
                            user_id=session.user_id,
                            decision_time=timestamp,
                            selected_candidate=None,
                            selected_artist=None,
                            selected_track=None,
                            selected_score=None,
                            selected_context=None,
                            selected_scaled_context=None,
                            reward=None,
                            reward_available_time=None,
                            candidate_count=0,
                            top_k_candidates=(),
                            top_k_scores=(),
                            top_k_effective_size=0,
                            linucb_mean=None,
                            linucb_exploration=None,
                            feedback_applied_before=feedback_applied,
                        )
                    )
                    skipped += 1
                    continue

                scaled_contexts = tuple(
                    self._scaler.transform(context) for context in raw_contexts
                )
                score_rows = self._learner.score_batch(
                    scaled_contexts,
                    candidate_ids=(candidate.track_key for candidate in pools.candidates),
                )
                top1 = max(score_rows, key=lambda item: (item.score, -item.index))
                top_k_rows = tuple(
                    sorted(score_rows, key=lambda item: (-item.score, item.index))[
                        : self._top_k
                    ]
                )
                selected = pools.candidates[top1.index]
                selected_raw_context = raw_contexts[top1.index]
                selected_scaled_context = scaled_contexts[top1.index]
                reward = self._reward_calculator.reward(
                    session.user_id,
                    selected.track.artist_key,
                    timestamp,
                    self._reward_horizon,
                )
                availability = self._schedule_feedback(
                    timestamp, selected_scaled_context, reward
                )
                feedback_scheduled += 1
                decisions.append(
                    DecisionRecord(
                        phase=phase,
                        user_id=session.user_id,
                        decision_time=timestamp,
                        selected_candidate=selected,
                        selected_artist=selected.track.artist_key,
                        selected_track=selected.track_key,
                        selected_score=top1.score,
                        selected_context=selected_raw_context,
                        selected_scaled_context=selected_scaled_context,
                        reward=reward,
                        reward_available_time=availability,
                        candidate_count=len(pools.candidates),
                        top_k_candidates=tuple(
                            pools.candidates[item.index] for item in top_k_rows
                        ),
                        top_k_scores=tuple(item.score for item in top_k_rows),
                        top_k_effective_size=len(top_k_rows),
                        linucb_mean=top1.mean,
                        linucb_exploration=top1.exploration,
                        feedback_applied_before=feedback_applied,
                    )
                )

            # Same-time decision points are all complete before boundary events
            # are admitted; the next later decision will observe those events.
            self._state.advance_to(self._after_inclusive_time(timestamp))

        # Mature feedback through the inclusive phase end without making an
        # additional action. This leaves no validation reward in the test model.
        feedback_applied += self._process_pending(end)
        if self._state.current_time is None or self._state.current_time <= end:
            self._state.advance_to(self._after_inclusive_time(end))
        self._last_phase = phase

        return SimulationResult(
            phase=phase,
            decisions=tuple(decisions),
            decision_points=len(decisions),
            skipped_decisions=skipped,
            feedback_scheduled=feedback_scheduled,
            feedback_applied=feedback_applied,
            feedback_pending_at_end=len(self._pending),
        )
