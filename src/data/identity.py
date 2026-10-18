"""Canonical, typed identities shared by the recommendation pipeline.

Keys use explicit type tags; item keys add a length-prefixed source field
before the artist key. This framing is collision-safe even when source text
contains separators or punctuation used by older ``name::...`` keys. Source
text is preserved exactly; whitespace-only values count as missing but are
never trimmed or otherwise normalized when present.
"""

from __future__ import annotations

from typing import Optional


def _present(value: Optional[str]) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_canonical_artist_key(value: str) -> bool:
    """Whether ``value`` is a serialized key emitted by this module."""
    return (
        isinstance(value, str)
        and (value.startswith("AI:") or value.startswith("AN:"))
        and bool(value[3:].strip())
    )


def is_canonical_item_key(value: str) -> bool:
    """Whether ``value`` is a serialized item key emitted by this module."""
    if not isinstance(value, str) or len(value) < 4:
        return False
    if not (value.startswith("TI") or value.startswith("TN")):
        return False
    separator = value.find(":", 2)
    if separator < 0:
        return False
    length_text = value[2:separator]
    if not length_text.isdigit():
        return False
    field_length = int(length_text)
    field_start = separator + 1
    field_end = field_start + field_length
    source_value = value[field_start:field_end]
    artist_key = value[field_end:]
    return (
        bool(source_value.strip())
        and len(source_value) == field_length
        and is_canonical_artist_key(artist_key)
    )


def resolve_artist_key(artist_id: Optional[str], artist_name: Optional[str]) -> str:
    """Return a typed artist-ID key, or an exact-name fallback key.

    A nonblank original artist ID takes precedence. If it is absent, the
    original artist name is used without case, whitespace, or Unicode
    normalization. Missing both is an explicit data error.
    """
    if artist_id is not None and not isinstance(artist_id, str):
        raise TypeError("artist_id must be a string or None")
    if artist_name is not None and not isinstance(artist_name, str):
        raise TypeError("artist_name must be a string or None")
    if _present(artist_id):
        return f"AI:{artist_id}"
    if _present(artist_name):
        return f"AN:{artist_name}"
    raise ValueError("artist identity requires a nonblank artist ID or name")


def resolve_item_key(
    track_id: Optional[str], artist_key: str, track_name: Optional[str]
) -> str:
    """Return a track item key composed with its artist identity.

    When available, the original track ID is paired with ``artist_key`` so
    that one source track ID attributed to multiple artists yields distinct
    recommendation items. Without a track ID, the exact title and artist key
    form the fallback. A track ID alone is sufficient even if its title is
    absent; without either a usable ID or title, an explicit error is raised.
    """
    if not isinstance(artist_key, str) or not is_canonical_artist_key(artist_key):
        raise ValueError("item identity requires a resolved artist_key")
    if track_id is not None and not isinstance(track_id, str):
        raise TypeError("track_id must be a string or None")
    if track_name is not None and not isinstance(track_name, str):
        raise TypeError("track_name must be a string or None")
    if _present(track_id):
        return f"TI{len(track_id)}:{track_id}{artist_key}"
    if _present(track_name):
        return f"TN{len(track_name)}:{track_name}{artist_key}"
    raise ValueError("item identity requires a nonblank track ID or title")


def resolve_event_identity_keys(
    artist_id: Optional[str],
    artist_name: Optional[str],
    track_id: Optional[str],
    track_name: Optional[str],
) -> tuple[str, str]:
    """Resolve and return ``(item_key, artist_key)`` from source columns."""
    artist_key = resolve_artist_key(artist_id, artist_name)
    item_key = resolve_item_key(track_id, artist_key, track_name)
    return item_key, artist_key
