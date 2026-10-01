"""Minimal sanity checks for the incremental temporal state."""

from datetime import datetime, timedelta
import unittest

from src.data.state import PlayEvent, TemporalState


def ts(value: str) -> datetime:
    return datetime.fromisoformat(f"2005-10-30T{value}")


class TemporalStateSanityTest(unittest.TestCase):
    def test_s1_to_s2_keeps_current_session_event_invisible(self) -> None:
        events = [
            PlayEvent("U1", ts("10:00:00"), "T1", "A1"),
            PlayEvent("U1", ts("10:02:00"), "T2", "A1"),
            PlayEvent("U1", ts("10:05:00"), "T3", "A2"),
            PlayEvent("U1", ts("10:20:00"), "T4", "A2"),
        ]
        state = TemporalState(events)

    # First decision point: no event at t=10:00 belongs to the prior state.
        self.assertEqual(state.advance_to(ts("10:00:00")), 0)
        self.assertEqual(state.processed_event_count, 0)
        self.assertEqual(state.get_user_event_count("U1"), 0)

    # At S2, exactly the three S1 events are visible; T4 begins at t=10:20.
        self.assertEqual(state.advance_to(ts("10:20:00")), 3)
        self.assertEqual(state.processed_event_count, 3)
        self.assertEqual(state.get_user_event_count("U1"), 3)
        self.assertEqual(state.get_user_track_count("U1", "T1"), 1)
        self.assertEqual(state.get_user_track_count("U1", "T2"), 1)
        self.assertEqual(state.get_user_track_count("U1", "T3"), 1)
        self.assertEqual(state.get_user_artist_count("U1", "A1"), 2)
        self.assertEqual(state.get_user_artist_count("U1", "A2"), 1)
        self.assertEqual(state.get_track_play_count("T1"), 1)
        self.assertEqual(state.get_track_user_count("T1"), 1)
        self.assertEqual(state.get_artist_play_count("A1"), 2)
        self.assertEqual(state.get_artist_user_count("A1"), 1)
        self.assertEqual(state.get_track_first_timestamp("T3"), ts("10:05:00"))
        self.assertTrue(state.is_track_novel("U1", "T4"))
        self.assertTrue(state.is_artist_novel("U1", "A3"))
        self.assertEqual(
            state.get_user_track_recency("U1", "T1"), timedelta(minutes=20)
        )
        self.assertEqual(state.get_track_age("T1"), timedelta(minutes=20))
        self.assertEqual(state.get_track_play_count("T4"), 0)

    # T4 becomes visible only after time moves strictly past its timestamp.
        self.assertEqual(state.advance_to(ts("10:21:00")), 1)
        self.assertEqual(state.get_user_track_count("U1", "T4"), 1)
        self.assertEqual(state.get_track_play_count("T4"), 1)
        self.assertFalse(state.is_artist_novel("U1", "A2"))
