"""Canonical identity semantics and cross-module regression checks."""

from datetime import datetime, timedelta
import math
import tempfile
import unittest
from pathlib import Path

from src.data.identity import (
    resolve_artist_key,
    resolve_event_identity_keys,
    resolve_item_key,
)
from src.data.state import PlayEvent, TemporalState
from src.evaluation.metrics import OfflineMetricsEvaluator
from src.experiments.dataset import load_lastfm_tsv
from src.recommender.policies import RandomPolicy
from src.simulation.candidate_generator import CandidateGenerator
from src.simulation.context import ContextBuilder
from src.simulation.reward import RewardCalculator
from src.simulation.scaler import FeatureScaler
from src.simulation.simulator import SequentialOfflineBanditSimulator


T = datetime(2008, 12, 20)
ARTIST_A = resolve_artist_key("artist-A", "Artist A")
ARTIST_B = resolve_artist_key("artist-B", "Artist B")
SHARED_TRACK_ID = "7aebbc02-17ab-4f94-b91d-ff164c89f1f8"


def item(track_id: str, artist_key: str, title: str = "Open Mic Night") -> str:
    return resolve_item_key(track_id, artist_key, title)


class CanonicalIdentityTests(unittest.TestCase):
    def test_same_original_track_id_with_two_artist_ids_is_two_items(self):
        first = resolve_event_identity_keys(
            "188711ed-c99b-439c-844a-ca831f63a727", "Mf Doom",
            SHARED_TRACK_ID, "Open Mic Night",
        )
        second = resolve_event_identity_keys(
            "2aa5bd46-43ba-4a09-9c7e-20b07036db51", "Viktor Vaughn",
            SHARED_TRACK_ID, "Open Mic Nite",
        )
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_same_track_and_artist_repeated_resolve_to_same_key(self):
        one = resolve_event_identity_keys("a", "Name", "t", "Title")
        two = resolve_event_identity_keys("a", "Other display name", "t", "Different title")
        self.assertEqual(one, two)

    def test_missing_track_id_falls_back_to_artist_key_and_exact_title(self):
        artist = resolve_artist_key("artist-id", None)
        self.assertEqual(
            resolve_item_key(None, artist, "Exact Title"),
            resolve_item_key(None, artist, "Exact Title"),
        )
        self.assertNotEqual(
            resolve_item_key(None, artist, "Exact Title"),
            resolve_item_key(None, artist, "exact title"),
        )

    def test_missing_artist_id_falls_back_to_exact_name(self):
        self.assertEqual(
            resolve_artist_key(None, "Artist Name"),
            "AN:Artist Name",
        )
        self.assertNotEqual(
            resolve_artist_key(None, "Artist Name"),
            resolve_artist_key(None, "artist name"),
        )
        self.assertNotEqual(
            resolve_artist_key(None, "Artist Name"),
            resolve_artist_key(None, " Artist Name"),
        )

    def test_required_identity_fields_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "artist identity"):
            resolve_artist_key("  ", "\t")
        with self.assertRaisesRegex(ValueError, "item identity"):
            resolve_event_identity_keys("artist-id", None, None, " ")

    def test_canonical_serialization_is_collision_safe_and_order_stable(self):
        left_artist = resolve_artist_key(None, "A::B")
        right_artist = resolve_artist_key(None, "A")
        left = resolve_item_key(None, left_artist, "C")
        right = resolve_item_key(None, right_artist, "B::C")
        self.assertNotEqual(left, right)
        self.assertEqual(
            resolve_event_identity_keys("id", "Name", "track", "Title"),
            resolve_event_identity_keys("id", "Renamed", "track", "Other"),
        )
        reserved_looking_id = resolve_artist_key("AI:Artist Name", None)
        self.assertEqual(reserved_looking_id, "AI:AI:Artist Name")
        self.assertNotEqual(reserved_looking_id, resolve_artist_key("Artist Name", None))
        self.assertNotEqual(
            resolve_item_key("TI1:xAI:Artist Name", reserved_looking_id, None),
            resolve_item_key("TI1:x", resolve_artist_key("Artist Name", None), None),
        )

    def test_generator_separates_real_mf_doom_and_viktor_vaughn_collision(self):
        doom_item, doom_artist = resolve_event_identity_keys(
            "188711ed-c99b-439c-844a-ca831f63a727", "Mf Doom",
            SHARED_TRACK_ID, "Open Mic Night",
        )
        viktor_item, viktor_artist = resolve_event_identity_keys(
            "2aa5bd46-43ba-4a09-9c7e-20b07036db51", "Viktor Vaughn",
            SHARED_TRACK_ID, "Open Mic Nite",
        )
        events = [
            PlayEvent("user_000031", datetime(2006, 12, 16, 23, 41, 32), doom_item, doom_artist),
            PlayEvent("user_000053", datetime(2008, 6, 16, 10, 26, 50), viktor_item, viktor_artist),
        ]
        state = TemporalState(events)
        generator = CandidateGenerator(state)
        decision = datetime(2008, 6, 17)
        state.advance_to(decision)
        candidates = generator.generate("new-user", decision).candidates
        self.assertEqual({candidate.item_key for candidate in candidates}, {doom_item, viktor_item})
        self.assertEqual({candidate.track.artist_key for candidate in candidates}, {doom_artist, viktor_artist})

    def test_duplicate_items_merge_but_artist_reward_stays_artist_level(self):
        shared_a = item("same-track", ARTIST_A)
        shared_b = item("same-track", ARTIST_B)
        events = [
            PlayEvent("u", T, shared_a, ARTIST_A),
            PlayEvent("u", T + timedelta(seconds=1), shared_a, ARTIST_A),
            PlayEvent("u", T + timedelta(seconds=2), shared_b, ARTIST_B),
            PlayEvent("u", T + timedelta(days=1), item("future-a", ARTIST_A), ARTIST_A),
            PlayEvent("u", T + timedelta(days=1), item("future-b", ARTIST_B), ARTIST_B),
        ]
        state = TemporalState(events)
        state.advance_to(T + timedelta(seconds=3))
        self.assertEqual(state.get_item_play_count(shared_a), 2)
        self.assertEqual(state.get_item_play_count(shared_b), 1)
        rewards = RewardCalculator(events)
        self.assertAlmostEqual(
            rewards.reward_for_artist_key("u", ARTIST_A, T + timedelta(seconds=3)),
            math.log(2),
        )
        # A queried horizon that includes the future artist events attributes
        # each reward through its item's own canonical artist key.
        self.assertAlmostEqual(
            rewards.reward_for_artist_key("u", ARTIST_A, T, timedelta(days=2)),
            math.log(3),
        )
        self.assertAlmostEqual(
            rewards.reward_for_artist_key("u", ARTIST_B, T, timedelta(days=2)),
            math.log(3),
        )

    def test_candidate_context_novelty_is_item_and_artist_level(self):
        first = item(SHARED_TRACK_ID, ARTIST_A)
        second = item(SHARED_TRACK_ID, ARTIST_B)
        state = TemporalState([
            PlayEvent("catalog", T, first, ARTIST_A),
            PlayEvent("catalog", T + timedelta(seconds=1), second, ARTIST_B),
        ])
        generator = CandidateGenerator(state)
        decision_time = T + timedelta(seconds=2)
        state.advance_to(decision_time)
        candidates = generator.generate("new-user", decision_time).candidates
        vectors = [ContextBuilder(state).build("new-user", row, decision_time) for row in candidates]
        self.assertEqual(len({row.item_key for row in candidates}), 2)
        self.assertTrue(all(vector[13] == 1.0 and vector[14] == 1.0 for vector in vectors))

    def test_event_order_does_not_change_item_or_artist_keys(self):
        records = [
            ("a", "Artist A", "same", "Title"),
            ("b", "Artist B", "same", "Title"),
        ]
        forward = {resolve_event_identity_keys(*row) for row in records}
        reverse = {resolve_event_identity_keys(*row) for row in reversed(records)}
        self.assertEqual(forward, reverse)

    def test_future_event_is_invisible_until_strictly_after_timestamp(self):
        future_item = item("later", ARTIST_A)
        state = TemporalState([PlayEvent("u", T, future_item, ARTIST_A)])
        generator = CandidateGenerator(state)
        state.advance_to(T)
        self.assertEqual(generator.generate("u", T).candidates, ())
        state.advance_to(T + timedelta(microseconds=1))
        self.assertTrue(generator.has_observed_item(future_item))

    def test_tsv_loader_to_simulator_preserves_composite_identity(self):
        rows = (
            "catalog\t2008-12-01T00:00:00Z\tartist-A\tArtist A\tseed-a\tSeed A\n"
            "catalog\t2008-12-02T00:00:00Z\tartist-B\tArtist B\tseed-b\tSeed B\n"
            "catalog\t2008-12-20T00:00:00Z\tartist-A\tArtist A\tshared\tSong A\n"
            "catalog\t2008-12-20T00:01:00Z\tartist-B\tArtist B\tshared\tSong B\n"
            "u1\t2009-01-01T00:00:00Z\tartist-A\tArtist A\tnew-a\tSong A\n"
            "u1\t2009-01-02T00:00:00Z\tartist-B\tArtist B\tnew-b\tSong B\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.tsv"
            source.write_text(rows, encoding="utf-8")
            sequence = load_lastfm_tsv(source, temporary_directory=directory)
            try:
                events = tuple(sequence)
            finally:
                sequence.close()
        self.assertNotEqual(events[0].item_key, events[1].item_key)
        state = TemporalState(events)
        generator = CandidateGenerator(state)
        simulator = SequentialOfflineBanditSimulator(
            events=events,
            eligible_users={"u1"},
            state=state,
            candidate_generator=generator,
            context_builder=ContextBuilder(state),
            scaler=FeatureScaler(),
            reward_calculator=RewardCalculator(events),
            policy=RandomPolicy(7),
            candidate_pool_depth=50,
            top_k=1,
        )
        simulator.prepare_warmup(datetime(2008, 12, 1), datetime(2008, 12, 31, 23, 59, 59))
        result = simulator.run_phase("validation", datetime(2009, 1, 1), datetime(2009, 1, 8))
        self.assertTrue(result.decisions)
        self.assertEqual(len(result.eligible_catalog), 4)
        self.assertEqual(len(result.eligible_artists), 2)
        metrics = OfflineMetricsEvaluator(reward_calculator=RewardCalculator(events)).evaluate(result, k_values=(1,))
        self.assertAlmostEqual(metrics["catalog_coverage@1"].value, 0.25)
        self.assertAlmostEqual(metrics["artist_coverage@1"].value, 0.5)


if __name__ == "__main__":
    unittest.main()
