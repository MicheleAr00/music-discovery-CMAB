"""Unit tests for artist-aware offline recommendation metrics."""

from datetime import datetime, timedelta
import math
import unittest

from src.data.state import PlayEvent
from src.simulation.candidate_generator import (
    Candidate,
    CandidateSource,
    TrackMetadata,
)
from src.simulation.simulator import DecisionRecord, SimulationResult
from src.evaluation.metrics import OfflineMetricsEvaluator
from src.simulation.reward import RewardCalculator


T = datetime(2009, 1, 1, 12, 0, 0)
HORIZON = timedelta(days=7)


def event(user: str, when: datetime, artist: str, track: str = "event-track"):
    return PlayEvent.from_raw(user, when, track, artist)


def item(track: str, artist: str) -> Candidate:
    return Candidate(
        TrackMetadata(track, track, artist, artist),
        frozenset({CandidateSource.POPULAR}),
    )


def decision(
    user: str,
    candidates: list[Candidate],
    *,
    when: datetime = T,
    phase: str = "validation",
    track_novelty: tuple[bool, ...] | None = None,
    artist_novelty: tuple[bool, ...] | None = None,
    candidate_count: int | None = None,
    reward: float | None = 0.0,
) -> DecisionRecord:
    if track_novelty is None:
        track_novelty = tuple(False for _ in candidates)
    if artist_novelty is None:
        artist_novelty = tuple(False for _ in candidates)
    selected = candidates[0] if candidates else None
    return DecisionRecord(
        phase=phase,
        user_id=user,
        decision_time=when,
        selected_candidate=selected,
        selected_artist=selected.track.artist_key if selected else None,
        selected_item_key=selected.item_key if selected else None,
        selected_score=0.0 if selected else None,
        selected_context=(0.0,) * 15 if selected else None,
        selected_scaled_context=(0.0,) * 15 if selected else None,
        reward=reward,
        reward_available_time=when + HORIZON if reward is not None else None,
        candidate_count=(len(candidates) if candidate_count is None else candidate_count),
        top_k_candidates=tuple(candidates),
        top_k_scores=tuple(0.0 for _ in candidates),
        top_k_effective_size=len(candidates),
        linucb_mean=0.0 if selected else None,
        linucb_exploration=0.0 if selected else None,
        feedback_applied_before=0,
        top_k_track_novelty=track_novelty,
        top_k_artist_novelty=artist_novelty,
    )


def result(
    decisions: list[DecisionRecord],
    *,
    phase: str = "validation",
    catalog: set[str] | None = None,
    artists: set[str] | None = None,
) -> SimulationResult:
    candidates = {
        candidate.track_key
        for record in decisions
        for candidate in record.top_k_candidates
    }
    artist_keys = {
        candidate.track.artist_key
        for record in decisions
        for candidate in record.top_k_candidates
    }
    return SimulationResult(
        phase=phase,
        decisions=tuple(decisions),
        decision_points=len(decisions),
        skipped_decisions=sum(not row.top_k_candidates for row in decisions),
        feedback_scheduled=sum(row.reward is not None for row in decisions),
        feedback_applied=0,
        feedback_pending_at_end=0,
        eligible_catalog=frozenset(candidates if catalog is None else catalog),
        eligible_artists=frozenset(artist_keys if artists is None else artists),
    )


class OfflineMetricsEvaluatorTests(unittest.TestCase):
    def metric(self, rows, events, name, *, phase="validation", k=5, **sets):
        evaluated = OfflineMetricsEvaluator(events).evaluate(
            result(rows, phase=phase, **sets), k_values=(k,)
        )
        return evaluated[name]

    def test_evaluator_can_share_the_simulators_read_only_future_index(self):
        rows = [decision("U", [item("T1", "A1")])]
        future = [event("U", T + timedelta(days=1), "A1")]
        simulation = result(rows, phase="validation")
        calculator = RewardCalculator(future, horizon=HORIZON)
        shared = OfflineMetricsEvaluator(
            horizon=HORIZON, reward_calculator=calculator
        ).evaluate(simulation, k_values=(5,))
        standalone = OfflineMetricsEvaluator(
            future, horizon=HORIZON
        ).evaluate(simulation, k_values=(5,))
        self.assertEqual(shared.metrics, standalone.metrics)
        with self.assertRaisesRegex(ValueError, "events or reward_calculator"):
            OfflineMetricsEvaluator(
                future, horizon=HORIZON, reward_calculator=calculator
            )

    def test_empty_candidate_decision_is_excluded_from_metric_aggregation(self):
        valid = decision("U1", [item("T1", "A1")], reward=0.75)
        empty = decision("U2", [], reward=None)
        events = [
            event("U1", T + timedelta(days=1), "A1"),
            event("U2", T + timedelta(days=1), "A2"),
        ]
        evaluator = OfflineMetricsEvaluator(events)
        only_valid = evaluator.evaluate(result([valid]), k_values=(5,))
        with_empty = evaluator.evaluate(result([valid, empty]), k_values=(5,))
        self.assertEqual(with_empty.decisions_evaluated, 1)
        self.assertEqual(with_empty.users_evaluated, 1)
        self.assertEqual(2 - with_empty.decisions_evaluated, 1)
        self.assertEqual(with_empty.metrics, only_valid.metrics)
        self.assertEqual(with_empty["precision@5"].value, 1.0)
        self.assertEqual(with_empty["recall@5"].value, 1.0)

    def test_precision_with_fully_relevant_list(self):
        rows = [decision("U", [item("T1", "A1"), item("T2", "A2")])]
        events = [event("U", T + timedelta(days=1), "A1"), event("U", T + timedelta(days=2), "A2")]
        self.assertEqual(self.metric(rows, events, "precision@5").value, 1.0)

    def test_precision_with_partially_relevant_list(self):
        rows = [decision("U", [item("T1", "A1"), item("T2", "A2")])]
        events = [event("U", T + timedelta(days=1), "A1")]
        self.assertEqual(self.metric(rows, events, "precision@5").value, 0.5)

    def test_recall_with_zero_denominator_is_undefined_and_excluded(self):
        evaluated = OfflineMetricsEvaluator([]).evaluate(
            result([decision("U", [item("T1", "A1")])]), k_values=(5,)
        )
        self.assertIsNone(evaluated["recall@5"].value)
        self.assertEqual(evaluated["recall@5"].decisions_used, 0)
        self.assertEqual(evaluated.recall_excluded_decisions, 1)

    def test_recall_is_artist_level_for_duplicate_tracks_by_one_artist(self):
        rows = [decision("U", [item("T1", "A1"), item("T2", "A1")])]
        events = [
            event("U", T + timedelta(days=1), "A1"),
            event("U", T + timedelta(days=1), "A2"),
        ]
        self.assertEqual(self.metric(rows, events, "recall@5").value, 0.5)

    def test_artist_relevance_ranking_deduplicates_in_first_occurrence_order(self):
        # Top tracks A1, A2, B1 become the ordered artist list A, B.
        rows = [decision("U", [item("A1", "A"), item("A2", "A"), item("B1", "B")])]
        events = [event("U", T + timedelta(days=1), "B")]
        metric = self.metric(rows, events, "ndcg@5")
        self.assertAlmostEqual(metric.value, 1 / math.log2(3))

    def test_precision_counts_distinct_recommended_artists_not_tracks(self):
        rows = [
            decision(
                "U",
                [item("A1", "A"), item("A2", "A"), item("B1", "B")],
            )
        ]
        events = [event("U", T + timedelta(days=1), "A")]
        self.assertEqual(self.metric(rows, events, "precision@5").value, 0.5)

    def test_recall_duplicate_artist_counts_once_in_numerator(self):
        rows = [decision("U", [item("A1", "A"), item("A2", "A"), item("B1", "B")])]
        events = [
            event("U", T + timedelta(days=1), "A"),
            event("U", T + timedelta(days=2), "B"),
            event("U", T + timedelta(days=3), "C"),
        ]
        self.assertAlmostEqual(self.metric(rows, events, "recall@5").value, 2 / 3)

    def test_ndcg_deduplicates_artist_gains_and_is_bounded_by_one(self):
        rows = [decision("U", [item("A1", "A"), item("A2", "A"), item("A3", "A")])]
        events = [event("U", T + timedelta(days=1), "A")]
        ndcg = self.metric(rows, events, "ndcg@5").value
        self.assertEqual(ndcg, 1.0)
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_all_top_k_tracks_from_one_artist_yield_one_artist_recommendation(self):
        rows = [decision("U", [item("A1", "A"), item("A2", "A"), item("A3", "A")])]
        events = [event("U", T + timedelta(days=1), "A")]
        evaluated = OfflineMetricsEvaluator(events).evaluate(result(rows), k_values=(5,))
        self.assertEqual(evaluated["precision@5"].value, 1.0)
        self.assertEqual(evaluated["recall@5"].value, 1.0)
        self.assertEqual(evaluated["ndcg@5"].value, 1.0)

    def test_ndcg_ideal_uses_all_relevant_artists_when_deduped_ranking_is_shorter(self):
        rows = [decision("U", [item("A1", "A"), item("A2", "A"), item("A3", "A")])]
        events = [
            event("U", T + timedelta(days=1), "A"),
            event("U", T + timedelta(days=2), "B"),
            event("U", T + timedelta(days=3), "C"),
        ]
        ndcg = self.metric(rows, events, "ndcg@5").value
        expected = 1.0 / sum(1.0 / math.log2(rank + 2) for rank in range(3))
        self.assertAlmostEqual(ndcg, expected)
        self.assertLess(ndcg, 1.0)

    def test_ndcg_is_one_for_ideal_order(self):
        rows = [decision("U", [item("R1", "A1"), item("R2", "A2"), item("X", "AX")])]
        events = [event("U", T + timedelta(days=1), "A1"), event("U", T + timedelta(days=2), "A2")]
        self.assertAlmostEqual(self.metric(rows, events, "ndcg@5").value, 1.0)

    def test_ndcg_for_inverse_order(self):
        rows = [decision("U", [item("X", "AX"), item("R1", "A1"), item("R2", "A2")])]
        events = [event("U", T + timedelta(days=1), "A1"), event("U", T + timedelta(days=2), "A2")]
        expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(self.metric(rows, events, "ndcg@5").value, expected)

    def test_track_novelty_uses_logged_predecision_track_flags(self):
        row = decision(
            "U", [item("T1", "A1"), item("T2", "A1"), item("T3", "A2")],
            track_novelty=(True, False, True),
        )
        self.assertAlmostEqual(self.metric([row], [], "track_novelty@5").value, 2 / 3)

    def test_artist_novelty_uses_logged_predecision_artist_flags(self):
        row = decision(
            "U", [item("T1", "A1"), item("T2", "A1"), item("T3", "A2")],
            artist_novelty=(True, False, False),
        )
        self.assertEqual(self.metric([row], [], "artist_novelty@5").value, 0.5)

    def test_ild_is_zero_when_all_items_share_artist(self):
        row = decision("U", [item("T1", "A1"), item("T2", "A1"), item("T3", "A1")])
        self.assertEqual(self.metric([row], [], "artist_ild@5").value, 0.0)

    def test_ild_is_one_when_all_items_have_different_artists(self):
        row = decision("U", [item("T1", "A1"), item("T2", "A2"), item("T3", "A3")])
        self.assertEqual(self.metric([row], [], "artist_ild@5").value, 1.0)

    def test_ild_of_single_item_is_zero(self):
        row = decision("U", [item("T1", "A1")])
        self.assertEqual(self.metric([row], [], "artist_ild@5").value, 0.0)

    def test_macro_average_per_user_prevents_heavy_user_dominance(self):
        rows = [decision("light", [item("T1", "A1")])]
        rows.extend(
            decision("heavy", [item(f"T{i}", "AX")], when=T + timedelta(minutes=i))
            for i in range(9)
        )
        events = [event("light", T + timedelta(days=1), "A1")]
        metric = self.metric(rows, events, "precision@5")
        self.assertEqual(metric.aggregation, "macro_user")
        self.assertEqual(metric.value, 0.5)
        self.assertEqual(metric.users_used, 2)
        self.assertEqual(metric.decisions_used, 10)

    def test_catalog_coverage_uses_known_eligible_catalog_denominator(self):
        rows = [decision("U", [item("T1", "A1"), item("T2", "A1")])]
        metric = self.metric(
            rows, [], "catalog_coverage@2", k=2,
            catalog={"T1", "T2", "T3", "T4"}, artists={"A1", "A2", "A3"},
        )
        self.assertEqual(metric.value, 0.5)
        self.assertEqual(metric.aggregation, "phase_union")

    def test_artist_coverage_counts_duplicate_artists_once(self):
        rows = [decision("U", [item("T1", "A1"), item("T2", "A1")])]
        metric = self.metric(
            rows, [], "artist_coverage@2", k=2,
            catalog={"T1", "T2"}, artists={"A1", "A2", "A3"},
        )
        self.assertAlmostEqual(metric.value, 1 / 3)

    def test_validation_and_test_are_evaluated_separately(self):
        validation = result([decision("U", [item("V", "AV")])])
        test = result(
            [decision("U", [item("T", "AT"),], phase="test")],
            phase="test",
        )
        evaluator = OfflineMetricsEvaluator([])
        validation_report = evaluator.evaluate(validation, k_values=(5,))
        test_report = evaluator.evaluate(test, k_values=(5,))
        self.assertEqual(validation_report.phase, "validation")
        self.assertEqual(test_report.phase, "test")
        self.assertEqual(validation_report["catalog_coverage"].value, 1.0)
        self.assertEqual(test_report["catalog_coverage"].value, 1.0)

    def test_k_larger_than_effective_list_uses_available_list_length(self):
        row = decision("U", [item("T1", "A1"), item("T2", "A2")])
        events = [event("U", T + timedelta(days=1), "A1")]
        metrics = self.metric([row], events, "precision@5", k=5)
        self.assertEqual(metrics.value, 0.5)
        self.assertEqual(metrics.decisions_used, 1)

    def test_evaluation_does_not_mutate_simulation_result(self):
        row = decision("U", [item("T1", "A1")])
        simulation = result([row])
        before = (simulation.decisions, simulation.eligible_catalog, simulation.eligible_artists)
        OfflineMetricsEvaluator([event("U", T + timedelta(days=1), "A1")]).evaluate(
            simulation, k_values=(5,)
        )
        after = (simulation.decisions, simulation.eligible_catalog, simulation.eligible_artists)
        self.assertEqual(before, after)

    def test_relevance_window_is_open_at_t_closed_at_t_plus_seven_days(self):
        rows = [decision("U", [item("AT", "AT"), item("END", "END"), item("LATE", "LATE")])]
        events = [
            event("U", T, "AT"),
            event("U", T + HORIZON, "END"),
            event("U", T + HORIZON + timedelta(microseconds=1), "LATE"),
        ]
        evaluated = OfflineMetricsEvaluator(events).evaluate(
            result(rows), k_values=(5,)
        )
        self.assertAlmostEqual(evaluated["precision@5"].value, 1 / 3)
        self.assertEqual(evaluated["recall@5"].value, 1.0)
        self.assertEqual(evaluated.recall_excluded_decisions, 0)


if __name__ == "__main__":
    unittest.main()
