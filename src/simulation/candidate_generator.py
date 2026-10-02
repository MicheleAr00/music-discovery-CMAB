"""Incremental temporal candidate pools for the music discovery simulator.

The generator is fed only by events accepted by :class:`TemporalState` while
advancing strictly before a decision point. It does not receive a full catalog
or any precomputed future statistics.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from src.data.state import PlayEvent, TemporalState


SUPPORTED_POOL_DEPTHS = frozenset({50, 100, 200})


def resolve_identity_key(identifier: Optional[str], name: Optional[str]) -> str:
    """Return an ID key or the established ``name::...`` fallback.

    Names are kept as supplied (including case); whitespace-only values count
    as missing. This applies no additional normalization or merging rule.
    """
    if identifier is not None and identifier.strip():
        return identifier
    if name is not None and name.strip():
        return f"name::{name}"
    raise ValueError("an entity requires a nonblank ID or name")


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    """Track and artist identity fields used by source pools.

    IDs take precedence. Missing IDs fall back to ``name::<name>``; a track
    name is never used as a bare identity key. Internally observed event keys
    are stored in the ID fields as already-resolved identity keys.
    """

    track_id: Optional[str]
    track_name: Optional[str]
    artist_id: Optional[str]
    artist_name: Optional[str]

    @property
    def track_key(self) -> str:
        """Canonical track key compatible with ``TemporalState`` queries."""
        return resolve_identity_key(self.track_id, self.track_name)

    @property
    def artist_key(self) -> str:
        """Canonical artist key compatible with ``TemporalState`` queries."""
        return resolve_identity_key(self.artist_id, self.artist_name)


class CandidateSource(str, Enum):
    """Candidate-generation source membership."""

    PERSONAL = "personal"
    POPULAR = "popular"
    DISCOVERY = "discovery"


class DiscoveryType(str, Enum):
    """Novel-track subtype determined from pre-decision user history."""

    FAMILIAR_ARTIST = "familiar_artist"
    NEW_ARTIST = "new_artist"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One deduplicated track and all source memberships that selected it."""

    track: TrackMetadata
    sources: frozenset[CandidateSource]
    discovery_type: Optional[DiscoveryType] = None

    @property
    def track_key(self) -> str:
        """Canonical key of the candidate track."""
        return self.track.track_key


@dataclass(frozen=True, slots=True)
class CandidatePools:
    """Independent top-N source pools and their ordered deduplicated union."""

    personal: tuple[Candidate, ...]
    popular: tuple[Candidate, ...]
    discovery: tuple[Candidate, ...]
    candidates: tuple[Candidate, ...]


class _IndexedPopularityHeap:
    """Mutable min-heap storing one current popularity entry per track."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, str]] = []
        self._positions: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._heap)

    def set_count(self, track_key: str, play_count: int) -> None:
        """Insert or update a track, ordered by count descending then key."""
        entry = (-play_count, track_key)
        position = self._positions.get(track_key)
        if position is None:
            self._heap.append(entry)
            position = len(self._heap) - 1
            self._positions[track_key] = position
            self._sift_up(position)
            return

        old_entry = self._heap[position]
        self._heap[position] = entry
        if entry < old_entry:
            self._sift_up(position)
        elif old_entry < entry:
            self._sift_down(position)

    def pop(self) -> tuple[int, str]:
        """Remove and return the most popular current entry."""
        if not self._heap:
            raise IndexError("pop from empty popularity heap")
        root = self._heap[0]
        last = self._heap.pop()
        del self._positions[root[1]]
        if self._heap:
            self._heap[0] = last
            self._positions[last[1]] = 0
            self._sift_down(0)
        return root

    def restore(self, entry: tuple[int, str]) -> None:
        """Reinsert an entry temporarily removed by a top-N query."""
        _, track_key = entry
        if track_key in self._positions:
            raise ValueError(f"track {track_key!r} is already in popularity heap")
        self._heap.append(entry)
        position = len(self._heap) - 1
        self._positions[track_key] = position
        self._sift_up(position)

    def _swap(self, left: int, right: int) -> None:
        self._heap[left], self._heap[right] = self._heap[right], self._heap[left]
        self._positions[self._heap[left][1]] = left
        self._positions[self._heap[right][1]] = right

    def _sift_up(self, position: int) -> None:
        while position:
            parent = (position - 1) // 2
            if self._heap[parent] <= self._heap[position]:
                break
            self._swap(parent, position)
            position = parent

    def _sift_down(self, position: int) -> None:
        size = len(self._heap)
        while True:
            left = 2 * position + 1
            right = left + 1
            smallest = position
            if left < size and self._heap[left] < self._heap[smallest]:
                smallest = left
            if right < size and self._heap[right] < self._heap[smallest]:
                smallest = right
            if smallest == position:
                return
            self._swap(position, smallest)
            position = smallest


class CandidateGenerator:
    """Maintain candidate indexes from the temporal state's consumed events.

    Construct this object before the first event is processed by ``state``.
    The state's event listener populates the observed track/artist mapping,
    artist-to-track index, discovery key heap, and popularity heap only when an
    event is consumed by ``advance_to(t)``. Since that method consumes only
    ``timestamp < t``, future items cannot enter these indexes.

    For ranking, popularity is ordered by descending observed play count and
    then ascending canonical track key. Personal and discovery pools use
    ascending canonical track key. These simple deterministic orders add no
    model score or exploration policy. Each source is independently truncated
    to ``pool_depth`` before the union is deduplicated.
    """

    def __init__(self, state: TemporalState) -> None:
        self._state = state
        self._track_artists: dict[str, str] = {}
        self._tracks_by_artist: dict[str, list[str]] = {}
        self._observed_keys_heap: list[str] = []
        self._popularity_heap = _IndexedPopularityHeap()
        self._closed = False
        state.add_event_listener(self._observe_event)

    @property
    def observed_track_count(self) -> int:
        """Number of unique tracks observed by the state so far."""
        return len(self._track_artists)

    @property
    def indexed_popularity_track_count(self) -> int:
        """Number of unique track entries in the popularity index."""
        return len(self._popularity_heap)

    def has_observed_track(self, track_key: str) -> bool:
        """Whether ``track_key`` has entered the state strictly before now."""
        return track_key in self._track_artists

    def close(self) -> None:
        """Detach from the state when this generator is no longer needed."""
        self._state.remove_event_listener(self._observe_event)
        self._closed = True

    def _observe_event(self, event: PlayEvent) -> None:
        track_key, artist_key = event.track_id, event.artist_id
        previous_artist = self._track_artists.get(track_key)
        if previous_artist is None:
            self._track_artists[track_key] = artist_key
            artist_heap = self._tracks_by_artist.setdefault(artist_key, [])
            heapq.heappush(artist_heap, track_key)
            heapq.heappush(self._observed_keys_heap, track_key)
        elif previous_artist != artist_key:
            raise ValueError(
                f"track key {track_key!r} is associated with multiple artists"
            )

        # The TemporalState applies the event before notifying listeners.
        plays = self._state.get_track_play_count(track_key)
        self._popularity_heap.set_count(track_key, plays)

    def _metadata_for(self, track_key: str) -> TrackMetadata:
        """Build the output record for a track already present in the index."""
        return TrackMetadata(
            track_id=track_key,
            track_name=None,
            artist_id=self._track_artists[track_key],
            artist_name=None,
        )

    def _top_popular_track_keys(self, limit: int) -> list[str]:
        """Read the popularity heap without consuming its current top entries."""
        selected_entries: list[tuple[int, str]] = []
        selected_keys: list[str] = []
        while len(self._popularity_heap) and len(selected_keys) < limit:
            entry = self._popularity_heap.pop()
            _, track_key = entry
            selected_entries.append(entry)
            selected_keys.append(track_key)
        for entry in selected_entries:
            self._popularity_heap.restore(entry)
        return selected_keys

    def _top_personal_track_keys(self, user_id: str, limit: int) -> list[str]:
        """Merge known artists' track-key heaps without scanning their catalog.

        Every track already played by the user belongs to an artist in that
        user's temporal artist history, so the known-artist index yields both
        required personal categories.
        """
        artist_heaps: dict[str, list[str]] = {}
        merge_heap: list[tuple[str, str]] = []
        temporarily_removed: dict[str, list[str]] = {}
        for artist_key in self._state.iter_user_artist_ids(user_id):
            artist_heap = self._tracks_by_artist.get(artist_key)
            if not artist_heap:
                continue
            artist_heaps[artist_key] = artist_heap
            first_key = heapq.heappop(artist_heap)
            temporarily_removed.setdefault(artist_key, []).append(first_key)
            heapq.heappush(merge_heap, (first_key, artist_key))

        selected: list[str] = []
        while merge_heap and len(selected) < limit:
            track_key, artist_key = heapq.heappop(merge_heap)
            selected.append(track_key)
            artist_heap = artist_heaps[artist_key]
            if artist_heap:
                next_key = heapq.heappop(artist_heap)
                temporarily_removed[artist_key].append(next_key)
                heapq.heappush(merge_heap, (next_key, artist_key))

        for artist_key, removed_keys in temporarily_removed.items():
            artist_heap = artist_heaps[artist_key]
            for track_key in removed_keys:
                heapq.heappush(artist_heap, track_key)
        return selected

    def _top_discovery_track_keys(self, user_id: str, limit: int) -> list[str]:
        """Return the first unseen keys, restoring the global heap afterward."""
        popped: list[str] = []
        selected: list[str] = []
        while self._observed_keys_heap and len(selected) < limit:
            track_key = heapq.heappop(self._observed_keys_heap)
            popped.append(track_key)
            if self._state.get_user_track_count(user_id, track_key) == 0:
                selected.append(track_key)
        for track_key in popped:
            heapq.heappush(self._observed_keys_heap, track_key)
        return selected

    def generate(
        self,
        user_id: str,
        timestamp: datetime,
        pool_depth: int = 50,
    ) -> CandidatePools:
        """Generate the three source pools and their union for ``(user,t)``.

        The state must be advanced to exactly ``timestamp`` before this call.
        The candidate indexes contain only events already consumed by that
        state. Personal candidates are the union of the user's observed tracks
        and observed tracks by artists in the user's history. Popular
        candidates use the incremental popularity heap; discovery candidates
        use the incremental observed-key heap and exclude user history.

        Args:
            user_id: User for whom candidates are generated.
            timestamp: Current decision point.
            pool_depth: Independent source-pool depth; 50, 100, or 200.

        Raises:
            ValueError: If the pool depth is unsupported or the state has not
                been advanced to the exact decision timestamp.
        """
        if pool_depth not in SUPPORTED_POOL_DEPTHS:
            raise ValueError(
                f"pool_depth must be one of {sorted(SUPPORTED_POOL_DEPTHS)}"
            )
        if self._closed:
            raise RuntimeError("CandidateGenerator has been closed")
        if self._state.current_time != timestamp:
            raise ValueError(
                "TemporalState must be advanced to the exact decision timestamp"
            )

        # A k-way merge visits only the user's known artists and at most N
        # selected tracks, instead of scanning the global track catalog.
        personal_keys = self._top_personal_track_keys(user_id, pool_depth)
        personal = tuple(
            Candidate(
                self._metadata_for(key),
                frozenset({CandidateSource.PERSONAL}),
            )
            for key in personal_keys
        )

        popular_keys = self._top_popular_track_keys(pool_depth)
        popular = tuple(
            Candidate(
                self._metadata_for(key),
                frozenset({CandidateSource.POPULAR}),
            )
            for key in popular_keys
        )

        discovery_keys = self._top_discovery_track_keys(user_id, pool_depth)
        discovery = tuple(
            Candidate(
                self._metadata_for(key),
                frozenset({CandidateSource.DISCOVERY}),
                (
                    DiscoveryType.FAMILIAR_ARTIST
                    if self._state.get_user_artist_count(
                        user_id, self._track_artists[key]
                    ) > 0
                    else DiscoveryType.NEW_ARTIST
                ),
            )
            for key in discovery_keys
        )

        merged: dict[str, Candidate] = {}
        for source_pool in (personal, popular, discovery):
            for candidate in source_pool:
                previous = merged.get(candidate.track_key)
                if previous is None:
                    merged[candidate.track_key] = candidate
                else:
                    merged[candidate.track_key] = Candidate(
                        track=previous.track,
                        sources=previous.sources | candidate.sources,
                        discovery_type=(
                            candidate.discovery_type
                            if candidate.discovery_type is not None
                            else previous.discovery_type
                        ),
                    )

        return CandidatePools(
            personal=personal,
            popular=popular,
            discovery=discovery,
            candidates=tuple(merged.values()),
        )
