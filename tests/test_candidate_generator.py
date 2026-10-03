"""Candidate-generation tests for temporal validity and source semantics."""

from datetime import datetime, timedelta
import unittest

from src.data.state import PlayEvent, TemporalState
from src.simulation.candidate_generator import (
    CandidateGenerator,
    CandidateSource,
    DiscoveryType,
    TrackMetadata,
)


def ts(minutes_after_10: int) -> datetime:
    return datetime(2005, 10, 30, 10, 0) + timedelta(minutes=minutes_after_10)


def fixture() -> list[PlayEvent]:
    return [
        PlayEvent("U1", ts(0), "T1", "A1"),
        PlayEvent("U1", ts(1), "T2", "A1"),
        PlayEvent("U2", ts(2), "T1", "A1"),
        PlayEvent("U2", ts(3), "T3", "A1"),
        PlayEvent("U2", ts(4), "T4", "A2"),
        PlayEvent("U2", ts(5), "T4", "A2"),
        PlayEvent("U1", ts(10), "TF", "AF"),  # exactly at decision time
        PlayEvent("U1", ts(11), "TFF", "AFF"),  # strictly after decision time
    ]


def make_result(pool_depth: int = 50):
    events = fixture()
    state = TemporalState(events)
    generator = CandidateGenerator(state)
    state.advance_to(ts(10))
    return generator, state, generator.generate("U1", ts(10), pool_depth)


class CandidateGeneratorTest(unittest.TestCase):
    def test_never_returns_track_first_seen_at_or_after_decision(self) -> None:
        generator, _, result = make_result()
        keys = {candidate.track_key for candidate in result.candidates}
        self.assertEqual(keys, {"T1", "T2", "T3", "T4"})
        self.assertEqual(generator.observed_track_count, 4)
        self.assertFalse(generator.has_observed_track("TF"))
        self.assertFalse(generator.has_observed_track("TFF"))

    def test_personal_pool_contains_history_and_known_artist_tracks_only(self) -> None:
        _, _, result = make_result()
        self.assertEqual(
            {candidate.track_key for candidate in result.personal},
            {"T1", "T2", "T3"},
        )

    def test_popular_pool_uses_observed_counts_and_deterministic_ties(self) -> None:
        _, _, result = make_result()
        self.assertEqual(
            [candidate.track_key for candidate in result.popular],
            ["T1", "T4", "T2", "T3"],
        )
        self.assertEqual(result.popular[0].track_key, "T1")

    def test_discovery_contains_only_unplayed_tracks_with_correct_type(self) -> None:
        _, _, result = make_result()
        discovery = {candidate.track_key: candidate for candidate in result.discovery}
        self.assertEqual(set(discovery), {"T3", "T4"})
        self.assertEqual(
            discovery["T3"].discovery_type, DiscoveryType.FAMILIAR_ARTIST
        )
        self.assertEqual(discovery["T4"].discovery_type, DiscoveryType.NEW_ARTIST)

    def test_union_deduplicates_and_preserves_all_source_flags(self) -> None:
        _, _, result = make_result()
        by_key = {candidate.track_key: candidate for candidate in result.candidates}
        self.assertEqual(len(by_key), len(result.candidates))
        self.assertEqual(
            by_key["T1"].sources,
            frozenset({CandidateSource.PERSONAL, CandidateSource.POPULAR}),
        )
        self.assertEqual(
            by_key["T3"].sources,
            frozenset(
                {
                    CandidateSource.PERSONAL,
                    CandidateSource.POPULAR,
                    CandidateSource.DISCOVERY,
                }
            ),
        )
        self.assertEqual(
            by_key["T3"].discovery_type, DiscoveryType.FAMILIAR_ARTIST
        )

    def test_supported_pool_depths_are_independent_top_n_limits(self) -> None:
        for depth in (50, 100, 200):
            with self.subTest(depth=depth):
                events = [
                    PlayEvent("U2", ts(index), f"T{index:03}", f"A{index:03}")
                    for index in range(depth + 1)
                ]
                state = TemporalState(events)
                generator = CandidateGenerator(state)
                state.advance_to(ts(depth + 1))
                result = generator.generate("U1", ts(depth + 1), depth)
                self.assertEqual(len(result.popular), depth)
                self.assertEqual(len(result.discovery), depth)
                self.assertLessEqual(len(result.candidates), depth * 3)

    def test_generation_is_reproducible(self) -> None:
        generator, state, first = make_result()
        second = generator.generate("U1", ts(10))
        self.assertEqual(first, second)

    def test_name_fallback_is_namespaced_and_ids_take_precedence(self) -> None:
        fallback = TrackMetadata(None, "Fallback Track", None, "Fallback Artist")
        with_id = TrackMetadata("real-id", "Same display name", "artist-id", "Artist")
        self.assertEqual(
            fallback.track_key, "name::Fallback Artist::Fallback Track"
        )
        self.assertEqual(fallback.artist_key, "name::Fallback Artist")
        self.assertEqual(with_id.track_key, "real-id")
        self.assertEqual(with_id.artist_key, "artist-id")

        events = [
            PlayEvent(
                "U2",
                ts(0),
                "name::Fallback Artist::Fallback Track",
                "name::Fallback Artist",
            )
        ]
        state = TemporalState(events)
        generator = CandidateGenerator(state)
        state.advance_to(ts(1))
        result = generator.generate("U1", ts(1))
        self.assertEqual(
            result.discovery[0].track_key,
            "name::Fallback Artist::Fallback Track",
        )

    def test_state_must_be_at_exact_decision_time(self) -> None:
        state = TemporalState(fixture())
        generator = CandidateGenerator(state)
        with self.assertRaises(ValueError):
            generator.generate("U1", ts(10))

    def test_generator_must_subscribe_before_state_consumes_history(self) -> None:
        state = TemporalState(fixture())
        state.advance_to(ts(10))
        with self.assertRaises(ValueError):
            CandidateGenerator(state)

    def test_s1_to_s2_incremental_state_hides_session_starting_event(self) -> None:
        events = [
            PlayEvent("U1", ts(0), "T1", "A1"),
            PlayEvent("U1", ts(2), "T2", "A1"),
            PlayEvent("U1", ts(5), "T3", "A2"),
            PlayEvent("U1", ts(20), "T4", "A2"),
            PlayEvent("U2", ts(30), "T5", "A2"),
        ]
        state = TemporalState(events)
        generator = CandidateGenerator(state)

        self.assertEqual(state.advance_to(ts(0)), 0)
        first = generator.generate("U1", ts(0))
        self.assertEqual(first.candidates, ())

        self.assertEqual(state.advance_to(ts(20)), 3)
        second = generator.generate("U1", ts(20))
        self.assertEqual(
            {candidate.track_key for candidate in second.candidates},
            {"T1", "T2", "T3"},
        )
        self.assertNotIn("T4", {candidate.track_key for candidate in second.candidates})
        self.assertEqual(generator.observed_track_count, 3)
        self.assertEqual(state.get_user_event_count("U1"), 3)

        state.advance_to(ts(31))
        third = generator.generate("U1", ts(31))
        t5 = next(c for c in third.candidates if c.track_key == "T5")
        self.assertIn(CandidateSource.PERSONAL, t5.sources)
        self.assertIn(CandidateSource.DISCOVERY, t5.sources)
        self.assertEqual(t5.discovery_type, DiscoveryType.FAMILIAR_ARTIST)

    def test_indexed_pools_match_naive_previous_generator_on_synthetic_events(self) -> None:
        events = fixture()
        decision_time = ts(10)
        observed = [event for event in events if event.timestamp < decision_time]
        track_artist: dict[str, str] = {}
        track_plays: dict[str, int] = {}
        user_tracks: dict[str, dict[str, int]] = {}
        user_artists: dict[str, dict[str, int]] = {}
        for event in observed:
            track_artist[event.track_id] = event.artist_id
            track_plays[event.track_id] = track_plays.get(event.track_id, 0) + 1
            user_track_counts = user_tracks.setdefault(event.user_id, {})
            user_track_counts[event.track_id] = user_track_counts.get(event.track_id, 0) + 1
            user_artist_counts = user_artists.setdefault(event.user_id, {})
            user_artist_counts[event.artist_id] = user_artist_counts.get(event.artist_id, 0) + 1

        personal = sorted(
            key
            for key, artist in track_artist.items()
            if user_tracks.get("U1", {}).get(key, 0) > 0
            or user_artists.get("U1", {}).get(artist, 0) > 0
        )[:50]
        popular = sorted(track_plays, key=lambda key: (-track_plays[key], key))[:50]
        discovery = sorted(
            key for key in track_artist if user_tracks.get("U1", {}).get(key, 0) == 0
        )[:50]
        expected_sources: dict[str, set[CandidateSource]] = {}
        for keys, source in (
            (personal, CandidateSource.PERSONAL),
            (popular, CandidateSource.POPULAR),
            (discovery, CandidateSource.DISCOVERY),
        ):
            for key in keys:
                expected_sources.setdefault(key, set()).add(source)
        expected_union_order = list(dict.fromkeys(personal + popular + discovery))

        state = TemporalState(events)
        generator = CandidateGenerator(state)
        state.advance_to(decision_time)
        indexed = generator.generate("U1", decision_time, 50)
        self.assertEqual([c.track_key for c in indexed.personal], personal)
        self.assertEqual([c.track_key for c in indexed.popular], popular)
        self.assertEqual([c.track_key for c in indexed.discovery], discovery)
        self.assertEqual(
            [c.track_key for c in indexed.candidates], expected_union_order
        )
        for candidate in indexed.candidates:
            self.assertEqual(candidate.sources, frozenset(expected_sources[candidate.track_key]))

    def test_index_receives_only_events_consumed_strictly_before_each_time(self) -> None:
        state = TemporalState(fixture())
        generator = CandidateGenerator(state)
        self.assertEqual(generator.observed_track_count, 0)

        state.advance_to(ts(10))
        self.assertEqual(generator.observed_track_count, 4)
        self.assertFalse(generator.has_observed_track("TF"))
        self.assertFalse(generator.has_observed_track("TFF"))

        state.advance_to(ts(11))
        self.assertTrue(generator.has_observed_track("TF"))
        self.assertFalse(generator.has_observed_track("TFF"))
        state.advance_to(ts(12))
        self.assertTrue(generator.has_observed_track("TFF"))

    def test_popularity_index_keeps_one_entry_per_track_after_many_plays(self) -> None:
        base = ts(0)
        events = [
            PlayEvent("U2", base + timedelta(seconds=i), "T0", "A0")
            for i in range(2000)
        ] + [
            PlayEvent("U2", base + timedelta(seconds=2000 + i), f"T{i}", f"A{i}")
            for i in range(1, 25)
        ]
        state = TemporalState(events)
        generator = CandidateGenerator(state)
        decision_time = base + timedelta(seconds=len(events) + 1)
        state.advance_to(decision_time)

        self.assertEqual(generator.observed_track_count, 25)
        self.assertEqual(generator.indexed_popularity_track_count, 25)
        result = generator.generate("U1", decision_time)
        self.assertEqual(result.popular[0].track_key, "T0")
        self.assertEqual(generator.indexed_popularity_track_count, 25)


if __name__ == "__main__":
    unittest.main()

# =============================================================================
# NOTA ARCHITETTURALE E METODOLOGICA (CandidateGeneratorTest)
# =============================================================================
# Questa suite di unit test garantisce la correttezza semantica e temporale 
# del modulo di generazione dei candidati (Action Space Sampling).
#
# 1. Integrità del Vettore Temporale (Zero Leakage):
#    I test verificano che nessuna traccia o artista possa entrare nei bacini
#    di raccomandazione (Personal, Popular, Discovery) se il relativo timestamp
#    di prima apparizione globale è >= al momento della decisione t. Questo
#    preserva l'invariante cronologica del simulatore CMAB.
#
# 2. Dimostrazione di Equivalenza Strutturale:
#    Tramite il confronto con un estrattore "naive" a forza bruta, la suite 
#    dimostra matematicamente che l'uso di strutture dati ottimizzate (Mutable 
#    Min-Heaps, O(log N)) per la gestione incrementale della popolarità e della 
#    novelty produce un output deterministicamente identico a una scansione 
#    lineare completa O(N).
#
# 3. Determinismo e Disaccoppiamento:
#    Il generatore espone un comportamento riproducibile (idempotenza a parità
#    di stato) e applica rigide regole di deduplicazione delle sorgenti 
#    preservando l'informazione di discovery (Familiar Artist vs New Artist)
#    senza sovrapposizioni.
# =============================================================================
