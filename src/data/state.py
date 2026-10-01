"""Incremental temporal state for pre-decision recommendation features.

The state consumes chronologically ordered listening events and exposes only
events whose timestamp is strictly earlier than the current decision time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import DefaultDict, Iterable, Iterator, Optional


@dataclass(frozen=True, slots=True)
class PlayEvent:
    """One observed listening event.

    ``timestamp`` must use a consistent timezone convention throughout a
    stream. Events must be supplied in nondecreasing timestamp order.
    """

    user_id: str
    timestamp: datetime
    track_id: str
    artist_id: str


class TemporalState:
    """Maintain user and global statistics as time advances.

    Call :meth:`advance_to` at each decision point before querying features.
    The method consumes only events with ``event.timestamp < timestamp``;
    events exactly at the decision timestamp remain invisible. Advancing is
    monotonic and each event is consumed at most once, so updates are O(1)
    amortized per event and queries are O(1).

    Args:
        events: Chronologically ordered listening events. The iterable is
            consumed lazily, which avoids rebuilding statistics at each
            decision point.
    """

    def __init__(self, events: Iterable[PlayEvent] = ()) -> None:
        self._events: Iterator[PlayEvent] = iter(events)
        self._next_event: Optional[PlayEvent] = next(self._events, None)
        self._last_consumed_timestamp: Optional[datetime] = None
        self._current_time: Optional[datetime] = None
        self._processed_event_count = 0

        self._user_event_count: DefaultDict[str, int] = defaultdict(int)
        self._user_track_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._user_artist_counts: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._user_track_last_timestamp: dict[tuple[str, str], datetime] = {}
        self._user_artist_last_timestamp: dict[tuple[str, str], datetime] = {}

        self._track_play_counts: DefaultDict[str, int] = defaultdict(int)
        self._track_user_counts: DefaultDict[str, int] = defaultdict(int)
        self._track_seen_users: DefaultDict[str, set[str]] = defaultdict(set)
        self._artist_play_counts: DefaultDict[str, int] = defaultdict(int)
        self._artist_user_counts: DefaultDict[str, int] = defaultdict(int)
        self._artist_seen_users: DefaultDict[str, set[str]] = defaultdict(set)
        self._track_first_timestamp: dict[str, datetime] = {}

    @property
    def current_time(self) -> Optional[datetime]:
        """Most recent decision time passed to :meth:`advance_to`, if any."""
        return self._current_time

    @property
    def processed_event_count(self) -> int:
        """Number of events incorporated into the observable state."""
        return self._processed_event_count

    def advance_to(self, timestamp: datetime) -> int:
        """Consume events strictly before ``timestamp`` and return their count.

        Repeating the same decision time is safe and consumes no additional
        events. Moving backwards raises ``ValueError`` because already
        consumed events cannot be removed from an incremental state.
        """
        if self._current_time is not None and timestamp < self._current_time:
            raise ValueError("TemporalState cannot advance to an earlier timestamp")

        consumed = 0
        while self._next_event is not None and self._next_event.timestamp < timestamp:
            event = self._next_event
            if (
                self._last_consumed_timestamp is not None
                and event.timestamp < self._last_consumed_timestamp
            ):
                raise ValueError("events must be ordered by nondecreasing timestamp")
            self._apply(event)
            self._processed_event_count += 1
            self._last_consumed_timestamp = event.timestamp
            self._next_event = next(self._events, None)
            consumed += 1

        # Check the next event too, including an event at the decision time.
        if (
            self._next_event is not None
            and self._last_consumed_timestamp is not None
            and self._next_event.timestamp < self._last_consumed_timestamp
        ):
            raise ValueError("events must be ordered by nondecreasing timestamp")
        self._current_time = timestamp
        return consumed

    def _apply(self, event: PlayEvent) -> None:
        user_id, track_id, artist_id = event.user_id, event.track_id, event.artist_id
        self._user_event_count[user_id] += 1
        self._user_track_counts[user_id][track_id] += 1
        self._user_artist_counts[user_id][artist_id] += 1
        self._user_track_last_timestamp[(user_id, track_id)] = event.timestamp
        self._user_artist_last_timestamp[(user_id, artist_id)] = event.timestamp

        self._track_play_counts[track_id] += 1
        if user_id not in self._track_seen_users[track_id]:
            self._track_seen_users[track_id].add(user_id)
            self._track_user_counts[track_id] += 1
        self._artist_play_counts[artist_id] += 1
        if user_id not in self._artist_seen_users[artist_id]:
            self._artist_seen_users[artist_id].add(user_id)
            self._artist_user_counts[artist_id] += 1
        self._track_first_timestamp.setdefault(track_id, event.timestamp)

    def get_user_event_count(self, user_id: str) -> int:
        """Return the number of pre-decision events for ``user_id``."""
        return self._user_event_count.get(user_id, 0)

    def get_user_track_count(self, user_id: str, track_id: str) -> int:
        """Return the user's pre-decision play count for a track."""
        return self._user_track_counts.get(user_id, {}).get(track_id, 0)

    def get_user_artist_count(self, user_id: str, artist_id: str) -> int:
        """Return the user's pre-decision play count for an artist."""
        return self._user_artist_counts.get(user_id, {}).get(artist_id, 0)

    def get_user_track_last_timestamp(
        self, user_id: str, track_id: str
    ) -> Optional[datetime]:
        """Return the last pre-decision play time, or ``None`` if unseen."""
        return self._user_track_last_timestamp.get((user_id, track_id))

    def get_user_artist_last_timestamp(
        self, user_id: str, artist_id: str
    ) -> Optional[datetime]:
        """Return the last pre-decision artist play time, or ``None`` if unseen."""
        return self._user_artist_last_timestamp.get((user_id, artist_id))

    def get_track_play_count(self, track_id: str) -> int:
        """Return global pre-decision plays for ``track_id``."""
        return self._track_play_counts.get(track_id, 0)

    def get_track_user_count(self, track_id: str) -> int:
        """Return distinct users with a pre-decision play of ``track_id``."""
        return self._track_user_counts.get(track_id, 0)

    def get_artist_play_count(self, artist_id: str) -> int:
        """Return global pre-decision plays for ``artist_id``."""
        return self._artist_play_counts.get(artist_id, 0)

    def get_artist_user_count(self, artist_id: str) -> int:
        """Return distinct users with a pre-decision play of ``artist_id``."""
        return self._artist_user_counts.get(artist_id, 0)

    def get_track_first_timestamp(self, track_id: str) -> Optional[datetime]:
        """Return the first observed pre-decision timestamp for a track."""
        return self._track_first_timestamp.get(track_id)

    def is_track_novel(self, user_id: str, track_id: str) -> bool:
        """Whether the user has not played this track before the decision."""
        return self.get_user_track_count(user_id, track_id) == 0

    def is_artist_novel(self, user_id: str, artist_id: str) -> bool:
        """Whether the user has not played this artist before the decision."""
        return self.get_user_artist_count(user_id, artist_id) == 0

    def get_user_track_recency(
        self, user_id: str, track_id: str, at: Optional[datetime] = None
    ) -> Optional[timedelta]:
        """Return elapsed time since the user's last play, or ``None`` unseen.

        If ``at`` is omitted, the current decision time is used. Recency is
        returned as a ``timedelta``; feature transformations can choose their
        units without changing this state module's temporal semantics.
        """
        last = self.get_user_track_last_timestamp(user_id, track_id)
        reference = at if at is not None else self._current_time
        if last is None or reference is None:
            return None
        if reference < last:
            raise ValueError("recency reference cannot precede the last play")
        return reference - last

    def get_user_artist_recency(
        self, user_id: str, artist_id: str, at: Optional[datetime] = None
    ) -> Optional[timedelta]:
        """Return elapsed time since the user's last artist play, if known."""
        last = self.get_user_artist_last_timestamp(user_id, artist_id)
        reference = at if at is not None else self._current_time
        if last is None or reference is None:
            return None
        if reference < last:
            raise ValueError("recency reference cannot precede the last play")
        return reference - last

    def get_track_age(
        self, track_id: str, at: Optional[datetime] = None
    ) -> Optional[timedelta]:
        """Return elapsed time since the track's first observed play, if known."""
        first = self.get_track_first_timestamp(track_id)
        reference = at if at is not None else self._current_time
        if first is None or reference is None:
            return None
        if reference < first:
            raise ValueError("age reference cannot precede the first play")
        return reference - first


# =============================================================================
# NOTA ARCHITETTURALE E METODOLOGICA (TemporalState)
# =============================================================================
# Questo modulo rappresenta il "Motore Temporale" del simulatore offline.
# La classe TemporalState è progettata per risolvere due problemi fondamentali
# della simulazione dei sistemi di raccomandazione sequenziali (CMAB):
#
# 1. Prevenzione del Data Leakage (Integrità Causale):
#    Il metodo `advance_to(timestamp)` fa scivolare l'orologio in avanti
#    consumando esclusivamente gli eventi strettamente precedenti al decision
#    point (D_{<t}). Qualsiasi evento al tempo `t` o nel futuro (D_{>=t}) 
#    rimane invisibile allo stato. Questo garantisce matematicamente che la 
#    generazione dei candidati e il calcolo del vettore di contesto non 
#    utilizzino mai informazioni future, emulando un vero ambiente online.
#
# 2. Efficienza Computazionale O(1):
#    Invece di ricalcolare metriche storiche (es. popolarità, history utente)
#    eseguendo costose query Pandas su milioni di righe a ogni raccomandazione,
#    lo stato mantiene dizionari globali e locali che si aggiornano in tempo
#    costante. L'interfaccia di lettura (get_*, is_*) permette a CandidateGenerator
#    e ContextBuilder di estrarre feature e condizioni di Novelty in O(1).
#
# Design Pattern:
# La classe attua una rigorosa "Separation of Concerns" (SoC). Gestisce 
# esclusivamente la contabilità del tempo e il tracciamento degli eventi, 
# delegando le trasformazioni matematiche (come logaritmi e standardizzazione) 
# al modulo ContextBuilder.
# =============================================================================