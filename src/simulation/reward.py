"""Indexed future artist-interaction reward for offline evaluation.

RewardCalculator intentionally has access to future events because it belongs
to the offline evaluation/simulation environment. The future information must
never be passed to the policy or observable state.
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
import math
from typing import Iterable, Optional

from src.data.identity import is_canonical_artist_key, resolve_artist_key
from src.data.state import PlayEvent


DEFAULT_REWARD_HORIZON = timedelta(days=7)
_MICROSECONDS_PER_SECOND = 1_000_000
_MICROSECONDS_PER_DAY = 86_400 * _MICROSECONDS_PER_SECOND
_EPOCH = datetime(1970, 1, 1)


def _is_aware(timestamp: datetime) -> bool:
    """Whether a datetime represents an aware instant."""
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _to_epoch_microseconds(timestamp: datetime) -> int:
    """Convert datetime to exact UTC microseconds without float rounding."""
    if _is_aware(timestamp):
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    delta = timestamp - _EPOCH
    return (
        delta.days * _MICROSECONDS_PER_DAY
        + delta.seconds * _MICROSECONDS_PER_SECOND
        + delta.microseconds
    )


def _horizon_microseconds(horizon: timedelta) -> int:
    """Validate and convert a positive timedelta to an integer duration."""
    if not isinstance(horizon, timedelta) or horizon <= timedelta(0):
        raise ValueError("horizon must be a positive timedelta")
    return (
        horizon.days * _MICROSECONDS_PER_DAY
        + horizon.seconds * _MICROSECONDS_PER_SECOND
        + horizon.microseconds
    )


class RewardCalculator:
    """Count future user-artist plays in ``(t, t + horizon]`` and log1p them.

    The supplied events are an offline future-data view, separate from the
    incrementally observable ``TemporalState``. Events must be chronological
    and carry the same resolved artist key used by the policy (ID when
    available, otherwise ``name::<artist_name>``). Timestamps are stored in
    compact sorted integer-microsecond arrays indexed by ``(user, artist)``.

    ``future_count`` costs O(log m), where m is the number of indexed events
    for that user-artist pair. ``future_artist_keys`` uses a separate compact
    per-user chronological event index and costs O(log E_u + H_u), where E_u
    is that user's event count and H_u is the number of events in the queried
    horizon. Both indexes are evaluation-only and isolated from policy state.
    """

    def __init__(
        self,
        events: Iterable[PlayEvent],
        horizon: timedelta = DEFAULT_REWARD_HORIZON,
    ) -> None:
        self._default_horizon = horizon
        _horizon_microseconds(horizon)
        self._timestamps: dict[tuple[str, str], array] = {}
        self._artist_indices: dict[str, int] = {}
        self._artist_keys: list[str] = []
        self._user_event_times: dict[str, array] = {}
        self._user_event_artist_indices: dict[str, array] = {}
        self._last_event_time: Optional[int] = None
        self._timezone_aware: Optional[bool] = None

        for event in events:
            self._validate_event(event)
            aware = _is_aware(event.timestamp)
            if self._timezone_aware is None:
                self._timezone_aware = aware
            elif aware != self._timezone_aware:
                raise ValueError("event timestamps must use a consistent timezone convention")

            event_time = _to_epoch_microseconds(event.timestamp)
            if self._last_event_time is not None and event_time < self._last_event_time:
                raise ValueError("future reward events must be ordered by timestamp")
            self._last_event_time = event_time

            key = (event.user_id, event.artist_key)
            times = self._timestamps.get(key)
            if times is None:
                times = array("q")
                self._timestamps[key] = times
            artist_index = self._artist_indices.get(event.artist_key)
            if artist_index is None:
                artist_index = len(self._artist_keys)
                self._artist_indices[event.artist_key] = artist_index
                self._artist_keys.append(event.artist_key)
            times.append(event_time)
            self._user_event_times.setdefault(event.user_id, array("q")).append(event_time)
            self._user_event_artist_indices.setdefault(event.user_id, array("I")).append(artist_index)

    @staticmethod
    def _validate_event(event: PlayEvent) -> None:
        if not isinstance(event, PlayEvent):
            raise TypeError("events must contain PlayEvent instances")
        if not isinstance(event.user_id, str) or not event.user_id.strip():
            raise ValueError("each event must have a nonblank user_id")
        if not isinstance(event.artist_key, str) or not event.artist_key.strip():
            raise ValueError("each event must have a nonblank resolved artist_key")
        if not isinstance(event.timestamp, datetime):
            raise ValueError("each event must have a datetime timestamp")

    def _resolve_query(
        self,
        user_id: str,
        artist_id: Optional[str],
        decision_time: datetime,
        horizon: Optional[timedelta],
        artist_name: Optional[str],
    ) -> tuple[tuple[str, str], int, int]:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a nonblank string")
        if not isinstance(decision_time, datetime):
            raise ValueError("decision_time must be a datetime")
        artist_key = resolve_artist_key(artist_id, artist_name)
        selected_horizon = horizon if horizon is not None else self._default_horizon
        horizon_us = _horizon_microseconds(selected_horizon)

        aware = _is_aware(decision_time)
        if self._timezone_aware is not None and aware != self._timezone_aware:
            raise ValueError(
                "decision_time must use the same timezone convention as indexed events"
            )
        start = _to_epoch_microseconds(decision_time)
        return (user_id, artist_key), start, start + horizon_us

    def future_count(
        self,
        user_id: str,
        artist_id: Optional[str],
        decision_time: datetime,
        horizon: Optional[timedelta] = None,
        *,
        artist_name: Optional[str] = None,
    ) -> int:
        """Count artist plays for this user in the half-open/closed horizon.

        The lower boundary is strict (events at ``decision_time`` are excluded)
        and the upper boundary is inclusive. Pass ``artist_name`` when the
        candidate has no artist ID; the existing project identity resolver then
        uses its canonical ``name::<artist_name>`` fallback.
        """
        key, start, end = self._resolve_query(
            user_id, artist_id, decision_time, horizon, artist_name
        )
        times = self._timestamps.get(key)
        if times is None:
            return 0
        return bisect_right(times, end) - bisect_right(times, start)

    def reward(
        self,
        user_id: str,
        artist_id: Optional[str],
        decision_time: datetime,
        horizon: Optional[timedelta] = None,
        *,
        artist_name: Optional[str] = None,
    ) -> float:
        """Return ``log(1 + future_count)`` for the requested user/artist."""
        count = self.future_count(
            user_id,
            artist_id,
            decision_time,
            horizon,
            artist_name=artist_name,
        )
        return math.log1p(count)

    def reward_for_artist_key(
        self,
        user_id: str,
        artist_key: str,
        decision_time: datetime,
        horizon: Optional[timedelta] = None,
    ) -> float:
        """Return reward for a canonical artist key already carried by an item."""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a nonblank string")
        if not isinstance(artist_key, str) or not is_canonical_artist_key(artist_key):
            raise ValueError("artist_key must be a canonical artist key")
        if not isinstance(decision_time, datetime):
            raise ValueError("decision_time must be a datetime")
        selected_horizon = horizon if horizon is not None else self._default_horizon
        horizon_us = _horizon_microseconds(selected_horizon)
        aware = _is_aware(decision_time)
        if self._timezone_aware is not None and aware != self._timezone_aware:
            raise ValueError("decision_time must use the same timezone convention as indexed events")
        times = self._timestamps.get((user_id, artist_key))
        if times is None:
            return 0.0
        start = _to_epoch_microseconds(decision_time)
        count = bisect_right(times, start + horizon_us) - bisect_right(times, start)
        return math.log1p(count)

    def future_artist_keys(
        self,
        user_id: str,
        decision_time: datetime,
        horizon: Optional[timedelta] = None,
    ) -> frozenset[str]:
        """Return distinct artists listened to in ``(t, t + horizon]``.

        The query bisects one chronological timestamp array for the user, then
        visits only events in the requested interval and deduplicates their
        compact artist ordinals. This avoids scanning every distinct artist
        the user has ever heard on each decision. The index is owned solely by
        the offline evaluator: it is not exposed to ``TemporalState``, context
        construction, or candidate generation, so it cannot affect policy
        observations or candidate pools.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a nonblank string")
        if not isinstance(decision_time, datetime):
            raise ValueError("decision_time must be a datetime")
        horizon_us = _horizon_microseconds(
            horizon if horizon is not None else self._default_horizon
        )
        aware = _is_aware(decision_time)
        if self._timezone_aware is not None and aware != self._timezone_aware:
            raise ValueError(
                "decision_time must use the same timezone convention as indexed events"
            )
        start = _to_epoch_microseconds(decision_time)
        end = start + horizon_us
        times = self._user_event_times.get(user_id)
        artist_indices = self._user_event_artist_indices.get(user_id)
        if times is None or artist_indices is None:
            return frozenset()
        first = bisect_right(times, start)
        stop = bisect_right(times, end)
        relevant_indices = {artist_indices[index] for index in range(first, stop)}
        return frozenset(self._artist_keys[index] for index in relevant_indices)
