"""Raw 15-dimensional, pre-decision contexts for candidate tracks."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TypeAlias

from src.data.state import TemporalState
from src.simulation.candidate_generator import Candidate


FEATURE_NAMES: tuple[str, ...] = (
    "USER_LOG_EVENTS",
    "USER_LOG_ARTISTS",
    "USER_LOG_TRACKS",
    "USER_ARTIST_REPETITION",
    "USER_TRACK_REPETITION",
    "TRACK_LOG_PLAYS",
    "TRACK_LOG_USERS",
    "ARTIST_LOG_PLAYS",
    "ARTIST_LOG_USERS",
    "TRACK_LOG_AGE",
    "USER_TRACK_LOG_COUNT",
    "USER_TRACK_LOG_RECENCY",
    "USER_ARTIST_LOG_COUNT",
    "TRACK_NOVELTY",
    "ARTIST_NOVELTY",
)
ContextVector: TypeAlias = tuple[float, ...]


def _seconds(elapsed: timedelta) -> float:
    """Convert duration to seconds, the documented raw time unit."""
    return elapsed.total_seconds()


class ContextBuilder:
    """Build raw contexts from a state already advanced to decision time.

    Time-based features use elapsed seconds before applying ``log1p``. This
    keeps the representation deterministic without imposing a dataset-specific
    time unit. The builder requires the candidate's identity keys to be the
    same resolved keys used by :class:`TemporalState`.
    """

    def __init__(self, state: TemporalState) -> None:
        self._state = state

    def build(
        self, user_id: str, candidate: Candidate, timestamp: datetime
    ) -> ContextVector:
        """Return the canonical 15 raw features for ``(user,candidate,t)``.

        ``TemporalState`` must already be at exactly ``timestamp``. Its
        ``advance_to`` method guarantees that only events strictly before the
        decision point have been incorporated. A candidate with no prior
        global observation is rejected, since candidate generation is defined
        over observed tracks only.

        Repetition ratios are defined as zero for a user with no prior events,
        avoiding division by zero while preserving a finite cold-start vector.
        """
        if self._state.current_time != timestamp:
            raise ValueError(
                "TemporalState must be advanced to the exact context timestamp"
            )

        track_key = candidate.track.track_key
        artist_key = candidate.track.artist_key
        first_track_time = self._state.get_track_first_timestamp(track_key)
        if first_track_time is None or first_track_time >= timestamp:
            raise ValueError(
                "candidate track must have been observed strictly before timestamp"
            )

        event_count = self._state.get_user_event_count(user_id)
        distinct_artist_count = self._state.get_user_distinct_artist_count(user_id)
        distinct_track_count = self._state.get_user_distinct_track_count(user_id)
        track_count = self._state.get_user_track_count(user_id, track_key)
        artist_count = self._state.get_user_artist_count(user_id, artist_key)

        age = self._state.get_track_age(track_key, timestamp)
        assert age is not None  # first_track_time above proves this exists.
        recency = self._state.get_user_track_recency(user_id, track_key, timestamp)

        artist_repetition = (
            1.0 - distinct_artist_count / event_count if event_count else 0.0
        )
        track_repetition = (
            1.0 - distinct_track_count / event_count if event_count else 0.0
        )
        recency_feature = (
            math.log1p(_seconds(recency))
            if track_count > 0 and recency is not None
            else 0.0
        )

        values = (
            math.log1p(event_count),
            math.log1p(distinct_artist_count),
            math.log1p(distinct_track_count),
            artist_repetition,
            track_repetition,
            math.log1p(self._state.get_track_play_count(track_key)),
            math.log1p(self._state.get_track_user_count(track_key)),
            math.log1p(self._state.get_artist_play_count(artist_key)),
            math.log1p(self._state.get_artist_user_count(artist_key)),
            math.log1p(_seconds(age)),
            math.log1p(track_count),
            recency_feature,
            math.log1p(artist_count),
            float(track_count == 0),
            float(artist_count == 0),
        )
        if len(values) != len(FEATURE_NAMES):
            raise RuntimeError("context feature schema and values are out of sync")
        return values
