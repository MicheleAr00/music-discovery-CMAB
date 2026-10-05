"""Memory-bounded TSV ingestion into a repeatable chronological event sequence.

The Last.fm source TSV is not guaranteed to be chronological (the bundled
file starts with its newest rows). This loader externally orders rows in a
temporary SQLite database, preserving source order among equal timestamps.
It does not load the 19M-event file into Python objects at once.
"""

from __future__ import annotations

import csv
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterator, Sequence, overload

from src.data.state import PlayEvent
from src.simulation.candidate_generator import (
    resolve_identity_key,
    resolve_track_identity_key,
)


_EPOCH = datetime(1970, 1, 1)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _microseconds(value: datetime) -> int:
    delta = value - _EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1_000_000) + delta.microseconds


class SQLiteEventSequence(Sequence[PlayEvent]):
    """Repeatable timestamp-ordered view over a temporary SQLite event table."""

    def __init__(self, database_path: Path, temporary_directory: tempfile.TemporaryDirectory[str]) -> None:
        self._database_path = database_path
        self._temporary_directory = temporary_directory
        self._closed = False

    def __len__(self) -> int:
        self._ensure_open()
        with closing(sqlite3.connect(self._database_path)) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def __iter__(self) -> Iterator[PlayEvent]:
        self._ensure_open()
        connection = sqlite3.connect(self._database_path)
        try:
            cursor = connection.execute(
                "SELECT user_id, timestamp_us, track_key, artist_key "
                "FROM events ORDER BY timestamp_us, ordinal"
            )
            for user_id, timestamp_us, track_key, artist_key in cursor:
                yield PlayEvent(
                    str(user_id),
                    _EPOCH + timedelta(microseconds=int(timestamp_us)),
                    str(track_key),
                    str(artist_key),
                )
        finally:
            connection.close()

    @overload
    def __getitem__(self, index: int) -> PlayEvent: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[PlayEvent]: ...

    def __getitem__(self, index: int | slice) -> PlayEvent | Sequence[PlayEvent]:
        self._ensure_open()
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return tuple(self[position] for position in range(start, stop, step))
            limit = max(0, stop - start)
            with closing(sqlite3.connect(self._database_path)) as connection:
                rows = connection.execute(
                    "SELECT user_id, timestamp_us, track_key, artist_key FROM events "
                    "ORDER BY timestamp_us, ordinal LIMIT ? OFFSET ?",
                    (limit, start),
                ).fetchall()
            return tuple(self._event(row) for row in rows)
        length = len(self)
        normalized = index + length if index < 0 else index
        if not 0 <= normalized < length:
            raise IndexError("event sequence index out of range")
        with closing(sqlite3.connect(self._database_path)) as connection:
            row = connection.execute(
                "SELECT user_id, timestamp_us, track_key, artist_key FROM events "
                "ORDER BY timestamp_us, ordinal LIMIT 1 OFFSET ?",
                (normalized,),
            ).fetchone()
        assert row is not None
        return self._event(row)

    @staticmethod
    def _event(row: tuple[object, ...]) -> PlayEvent:
        user_id, timestamp_us, track_key, artist_key = row
        return PlayEvent(
            str(user_id),
            _EPOCH + timedelta(microseconds=int(timestamp_us)),
            str(track_key),
            str(artist_key),
        )

    @property
    def database_path(self) -> Path:
        """Temporary sorted-event cache path, useful for diagnostics."""
        return self._database_path

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("event sequence is closed")

    def close(self) -> None:
        """Delete the temporary chronological cache and invalidate this view."""
        if not self._closed:
            self._closed = True
            self._temporary_directory.cleanup()

    def __enter__(self) -> SQLiteEventSequence:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def load_lastfm_tsv(
    path: str | Path,
    *,
    batch_size: int = 50_000,
    temporary_directory: str | Path | None = None,
) -> SQLiteEventSequence:
    """Load the six-column Last.fm 1K TSV into a disk-backed sorted sequence.

    The source has no header and columns are user, timestamp, artist ID,
    artist name, track ID, and track name. Missing IDs use the canonical
    identity fallback already shared by temporal state and candidate
    generation. Timestamps are normalized to naive UTC, matching the
    simulator's frozen date bounds.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    temp = tempfile.TemporaryDirectory(
        prefix="music_discovery_events_",
        dir=str(temporary_directory) if temporary_directory is not None else None,
    )
    database_path = Path(temp.name) / "chronological-events.sqlite3"
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE events ("
                "ordinal INTEGER PRIMARY KEY, timestamp_us INTEGER NOT NULL, "
                "user_id TEXT NOT NULL, track_key TEXT NOT NULL, artist_key TEXT NOT NULL)"
            )
            connection.execute("BEGIN")
            batch: list[tuple[int, int, str, str, str]] = []
            with source.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                reader = csv.reader(stream, delimiter="\t")
                for ordinal, row in enumerate(reader):
                    if len(row) != 6:
                        raise ValueError(f"TSV row {ordinal + 1} has {len(row)} columns; expected 6")
                    user_id, timestamp_text, artist_id, artist_name, track_id, track_name = row
                    user_id = user_id.strip()
                    if not user_id:
                        raise ValueError(f"TSV row {ordinal + 1} has a blank user ID")
                    timestamp_us = _microseconds(_parse_timestamp(timestamp_text))
                    artist_key = resolve_identity_key(artist_id or None, artist_name or None)
                    track_key = resolve_track_identity_key(
                        track_id or None, artist_name or None, track_name or None
                    )
                    batch.append((ordinal, timestamp_us, user_id, track_key, artist_key))
                    if len(batch) >= batch_size:
                        connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", batch)
                        batch.clear()
            if batch:
                connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", batch)
            connection.execute("CREATE INDEX events_chronological ON events(timestamp_us, ordinal)")
            connection.commit()
        return SQLiteEventSequence(database_path, temp)
    except BaseException:
        temp.cleanup()
        raise
