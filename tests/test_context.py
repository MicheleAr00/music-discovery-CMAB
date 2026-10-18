"""Tests for the raw, pre-decision 15-feature context builder."""

from datetime import datetime, timedelta
import math
import unittest

from src.data.state import PlayEvent, TemporalState
from src.data.identity import resolve_artist_key, resolve_item_key
from src.simulation.candidate_generator import (
    Candidate,
    CandidateGenerator,
    CandidateSource,
    TrackMetadata,
)
from src.simulation.context import ContextBuilder, FEATURE_NAMES


BASE = datetime(2005, 10, 30, 10, 0)


def ts(seconds: int) -> datetime:
    return BASE + timedelta(seconds=seconds)


def candidate(track: str = "T1", artist: str = "A1") -> Candidate:
    return Candidate(
        TrackMetadata(track, None, artist, None),
        frozenset({CandidateSource.POPULAR}),
    )


def known_history() -> list[PlayEvent]:
    return [
        PlayEvent.from_raw("U1", ts(0), "T1", "A1"),
        PlayEvent.from_raw("U1", ts(10), "T1", "A1"),
        PlayEvent.from_raw("U1", ts(20), "T2", "A1"),
        PlayEvent.from_raw("U2", ts(30), "T1", "A1"),
        PlayEvent.from_raw("U2", ts(40), "Tnew", "A1"),
        PlayEvent.from_raw("U2", ts(50), "T3", "A2"),
    ]


class ContextBuilderTest(unittest.TestCase):
    def make_context(self, track: str = "T1", artist: str = "A1"):
        state = TemporalState(known_history())
        state.advance_to(ts(60))
        return state, ContextBuilder(state).build("U1", candidate(track, artist), ts(60))

    def test_vector_has_exactly_15_numeric_features(self) -> None:
        _, values = self.make_context()
        self.assertEqual(len(values), 15)
        self.assertTrue(all(isinstance(value, float) for value in values))
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_feature_names_have_canonical_order(self) -> None:
        self.assertEqual(
            FEATURE_NAMES,
            (
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
            ),
        )

    def test_user_level_features_match_known_history(self) -> None:
        _, values = self.make_context()
        self.assertAlmostEqual(values[0], math.log1p(3))
        self.assertAlmostEqual(values[1], math.log1p(1))
        self.assertAlmostEqual(values[2], math.log1p(2))
        self.assertAlmostEqual(values[3], 1 - 1 / 3)
        self.assertAlmostEqual(values[4], 1 - 2 / 3)

    def test_global_track_features_and_age(self) -> None:
        _, values = self.make_context()
        self.assertAlmostEqual(values[5], math.log1p(3))
        self.assertAlmostEqual(values[6], math.log1p(2))
        self.assertAlmostEqual(values[9], math.log1p(60))

    def test_global_artist_features(self) -> None:
        _, values = self.make_context()
        self.assertAlmostEqual(values[7], math.log1p(5))
        self.assertAlmostEqual(values[8], math.log1p(2))

    def test_familiar_track_uses_exact_recency_and_zero_novelty(self) -> None:
        _, values = self.make_context()
        self.assertAlmostEqual(values[10], math.log1p(2))
        self.assertAlmostEqual(values[11], math.log1p(50))
        self.assertAlmostEqual(values[12], math.log1p(3))
        self.assertEqual(values[13:15], (0.0, 0.0))

    def test_new_track_with_familiar_artist(self) -> None:
        _, values = self.make_context("Tnew", "A1")
        self.assertEqual(values[10], 0.0)
        self.assertEqual(values[11], 0.0)
        self.assertEqual(values[12], math.log1p(3))
        self.assertEqual(values[13:15], (1.0, 0.0))

    def test_new_track_with_new_artist(self) -> None:
        _, values = self.make_context("T3", "A2")
        self.assertEqual(values[10:13], (0.0, 0.0, 0.0))
        self.assertEqual(values[13:15], (1.0, 1.0))

    def test_events_at_and_after_decision_do_not_leak_into_context(self) -> None:
        events = [
            PlayEvent.from_raw("U1", ts(0), "T1", "A1"),
            PlayEvent.from_raw("U1", ts(10), "T1", "A1"),  # exactly at decision
            PlayEvent.from_raw("U1", ts(20), "T2", "A2"),  # later than decision
        ]
        state = TemporalState(events)
        state.advance_to(ts(10))
        builder = ContextBuilder(state)
        values = builder.build("U1", candidate("T1", "A1"), ts(10))
        self.assertEqual(state.processed_event_count, 1)
        self.assertEqual(values[0], math.log1p(1))
        self.assertEqual(values[5], math.log1p(1))
        self.assertEqual(values[7], math.log1p(1))
        self.assertEqual(values[9], math.log1p(10))
        self.assertEqual(values[11], math.log1p(10))
        self.assertEqual(values[13:15], (0.0, 0.0))

    def test_context_changes_after_state_advances_past_new_events(self) -> None:
        events = [
            PlayEvent.from_raw("U1", ts(0), "T1", "A1"),
            PlayEvent.from_raw("U1", ts(10), "T1", "A1"),
            PlayEvent.from_raw("U1", ts(20), "T2", "A2"),
        ]
        state = TemporalState(events)
        builder = ContextBuilder(state)
        state.advance_to(ts(10))
        before = builder.build("U1", candidate("T1", "A1"), ts(10))
        state.advance_to(ts(21))
        after = builder.build("U1", candidate("T1", "A1"), ts(21))
        self.assertEqual(after[0], math.log1p(3))
        self.assertEqual(after[5], math.log1p(2))
        self.assertEqual(after[11], math.log1p(11))
        self.assertEqual(after[13:15], (0.0, 0.0))
        self.assertNotEqual(before, after)

    def test_identity_fallback_keys_match_state_queries(self) -> None:
        track_key = resolve_item_key(None, resolve_artist_key(None, "Fallback Artist"), "Fallback Track")
        artist_key = resolve_artist_key(None, "Fallback Artist")
        state = TemporalState(
            [PlayEvent("U2", ts(0), track_key, artist_key)]
        )
        state.advance_to(ts(5))
        metadata = TrackMetadata(
            None, "Fallback Track", None, "Fallback Artist"
        )
        self.assertEqual(metadata.track_key, track_key)
        self.assertEqual(metadata.artist_key, artist_key)
        values = ContextBuilder(state).build(
            "U1", Candidate(metadata, frozenset({CandidateSource.DISCOVERY})), ts(5)
        )
        self.assertEqual(values[5], math.log1p(1))
        self.assertEqual(values[7], math.log1p(1))
        self.assertEqual(values[13:15], (1.0, 1.0))

    def test_context_is_deterministic(self) -> None:
        state, first = self.make_context()
        second = ContextBuilder(state).build("U1", candidate("T1", "A1"), ts(60))
        self.assertEqual(first, second)

    def test_cold_start_single_play_and_new_candidate_edges(self) -> None:
        state = TemporalState([PlayEvent.from_raw("U2", ts(0), "T1", "A1")])
        state.advance_to(ts(1))
        values = ContextBuilder(state).build("U1", candidate("T1", "A1"), ts(1))
        self.assertEqual(values[:5], (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(values[5:9], (math.log1p(1),) * 4)
        self.assertEqual(values[9], math.log1p(1))
        self.assertEqual(values[10:13], (0.0, 0.0, 0.0))
        self.assertEqual(values[13:15], (1.0, 1.0))

    def test_rejects_state_not_at_decision_time(self) -> None:
        state = TemporalState(known_history())
        with self.assertRaises(ValueError):
            ContextBuilder(state).build("U1", candidate(), ts(60))

    def test_rejects_candidate_not_observed_before_decision(self) -> None:
        state = TemporalState([PlayEvent.from_raw("U2", ts(10), "T1", "A1")])
        state.advance_to(ts(10))
        with self.assertRaises(ValueError):
            ContextBuilder(state).build("U1", candidate(), ts(10))

    def test_candidate_generator_to_context_integration(self) -> None:
        events = known_history()
        state = TemporalState(events)
        generator = CandidateGenerator(state)
        state.advance_to(ts(60))
        pools = generator.generate("U1", ts(60))
        generated = next(item for item in pools.candidates if item.item_key == candidate("T1", "A1").item_key)
        values = ContextBuilder(state).build("U1", generated, ts(60))
        self.assertEqual(len(values), 15)
        self.assertEqual(values[5], math.log1p(state.get_item_play_count(candidate("T1", "A1").item_key)))
        self.assertEqual(values[13], 0.0)


if __name__ == "__main__":
    unittest.main()
