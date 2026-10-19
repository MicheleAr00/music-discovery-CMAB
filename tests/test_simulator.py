"""Synthetic tests for strict, delayed-feedback offline simulation."""

from datetime import datetime, timedelta
import math
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.data.state import PlayEvent, TemporalState
from src.data.identity import resolve_artist_key, resolve_item_key
from src.simulation.candidate_generator import CandidateGenerator, CandidatePools
from src.simulation.context import ContextBuilder
from src.simulation.linucb import LinUCB
from src.simulation.reward import RewardCalculator
from src.simulation.scaler import FeatureScaler
from src.simulation.simulator import (
    SequentialOfflineBanditSimulator,
    iter_session_starts,
)


BASE = datetime(2005, 10, 30)
VALIDATION_START = datetime(2009, 1, 1)
TEST_START = datetime(2009, 3, 1)
HORIZON = timedelta(days=7)


def event(user: str, when: datetime, track: str, artist: str = "A1") -> PlayEvent:
    return PlayEvent.from_raw(user, when, track, artist)


def make_events() -> list[PlayEvent]:
    return [
        event("U1", BASE, "T0"),
        event("U1", BASE + timedelta(minutes=20), "T1"),
        event("U1", BASE + timedelta(minutes=40), "T2"),
        event("U1", VALIDATION_START, "V0"),  # decision boundary
        event("U2", VALIDATION_START, "U2V0", "A2"),  # ineligible user
        event("U1", VALIDATION_START + timedelta(seconds=1), "V1"),
        event("U1", VALIDATION_START + timedelta(days=1), "V2"),
        event("U1", VALIDATION_START + timedelta(days=1, seconds=1), "V3"),
        event("U1", VALIDATION_START + HORIZON, "V4"),  # exact reward boundary
        event("U1", VALIDATION_START + HORIZON + timedelta(seconds=1), "V5"),
        event("U1", VALIDATION_START + timedelta(days=10), "VLATE"),
        event("U1", VALIDATION_START + timedelta(days=14), "VEND"),
        event("U1", TEST_START, "TEST0"),
        event("U1", TEST_START + timedelta(seconds=1), "TEST1"),
        event("U1", TEST_START + HORIZON, "TEST2"),
        event("U1", TEST_START + HORIZON + timedelta(seconds=1), "TEST3"),
        event("U1", TEST_START + timedelta(days=14), "TESTEND"),
    ]


def build_simulator(
    events: list[PlayEvent] | None = None,
    eligible_users: set[str] | None = None,
    *,
    scaler: FeatureScaler | None = None,
    learner: LinUCB | None = None,
    top_k: int = 10,
):
    source = make_events() if events is None else events
    state = TemporalState(source)
    generator = CandidateGenerator(state)
    scaler = FeatureScaler() if scaler is None else scaler
    simulator = SequentialOfflineBanditSimulator(
        events=source,
        eligible_users={"U1"} if eligible_users is None else eligible_users,
        state=state,
        candidate_generator=generator,
        context_builder=ContextBuilder(state),
        scaler=scaler,
        learner=LinUCB(alpha=0.2) if learner is None else learner,
        reward_calculator=RewardCalculator(source, horizon=HORIZON),
        candidate_pool_depth=50,
        top_k=top_k,
        reward_horizon=HORIZON,
    )
    return simulator


def prepare_default_warmup(simulator: SequentialOfflineBanditSimulator):
    return simulator.prepare_warmup(
        BASE - timedelta(seconds=1), BASE + timedelta(days=1)
    )


class SequentialSimulatorTest(unittest.TestCase):
    def test_precomputed_warmup_restores_equivalent_state_and_contexts(self) -> None:
        source = make_events()
        start = BASE - timedelta(seconds=1)
        end = BASE + timedelta(days=1)

        reference = build_simulator(source)
        reference_warmup = reference.prepare_warmup(start, end)

        restored = build_simulator(source)
        with (
            patch("src.simulation.simulator.iter_session_starts", side_effect=AssertionError),
            patch.object(restored._candidate_generator, "generate", side_effect=AssertionError),
            patch.object(restored._context_builder, "build", side_effect=AssertionError),
        ):
            restored_warmup = restored.prepare_warmup_from_precomputed(
                start,
                end,
                scaler_mean=reference.scaler.mean_,
                scaler_scale=reference.scaler.scale_,
                expected=reference_warmup,
            )

        self.assertEqual(reference_warmup, restored_warmup)
        self.assertIsNot(reference.state, restored.state)
        self.assertIsNot(reference.scaler, restored.scaler)
        self.assertIsNot(reference.scaler.mean_, restored.scaler.mean_)
        self.assertIsNot(reference.scaler.scale_, restored.scaler.scale_)
        self.assertEqual(reference.scaler.mean_, restored.scaler.mean_)
        self.assertEqual(reference.scaler.scale_, restored.scaler.scale_)
        self.assertEqual(reference.state.current_time, restored.state.current_time)
        self.assertEqual(
            reference.state.processed_event_count,
            restored.state.processed_event_count,
        )
        item_keys = {row.item_key for row in source}
        artist_keys = {row.artist_key for row in source}
        for user_id in {row.user_id for row in source}:
            self.assertEqual(
                reference.state.get_user_event_count(user_id),
                restored.state.get_user_event_count(user_id),
            )
            for item_key in item_keys:
                self.assertEqual(
                    reference.state.get_user_item_count(user_id, item_key),
                    restored.state.get_user_item_count(user_id, item_key),
                )
                self.assertEqual(
                    reference.state.get_user_track_last_timestamp(user_id, item_key),
                    restored.state.get_user_track_last_timestamp(user_id, item_key),
                )
                self.assertEqual(
                    reference.state.get_item_play_count(item_key),
                    restored.state.get_item_play_count(item_key),
                )
                self.assertEqual(
                    reference.state.get_item_user_count(item_key),
                    restored.state.get_item_user_count(item_key),
                )
                self.assertEqual(
                    reference.state.get_item_first_timestamp(item_key),
                    restored.state.get_item_first_timestamp(item_key),
                )
            for artist_key in artist_keys:
                self.assertEqual(
                    reference.state.get_user_artist_count(user_id, artist_key),
                    restored.state.get_user_artist_count(user_id, artist_key),
                )
                self.assertEqual(
                    reference.state.get_user_artist_last_timestamp(user_id, artist_key),
                    restored.state.get_user_artist_last_timestamp(user_id, artist_key),
                )
                self.assertEqual(
                    reference.state.get_artist_play_count(artist_key),
                    restored.state.get_artist_play_count(artist_key),
                )
                self.assertEqual(
                    reference.state.get_artist_user_count(artist_key),
                    restored.state.get_artist_user_count(artist_key),
                )

        # Both simultaneous session starts at validation time remain excluded
        # until their decision block has completed in the phase runner.
        reference.state.advance_to(VALIDATION_START)
        restored.state.advance_to(VALIDATION_START)
        simultaneous_u1_item = next(
            row.item_key for row in source
            if row.timestamp == VALIDATION_START and row.user_id == "U1"
        )
        for simulator in (reference, restored):
            self.assertEqual(
                simulator.state.get_user_item_count("U1", simultaneous_u1_item),
                0,
            )
        reference_generator = reference._candidate_generator
        restored_generator = restored._candidate_generator
        self.assertIsNot(reference_generator, restored_generator)
        for user_id in ("U1", "U2"):
            reference_pools = reference_generator.generate(user_id, VALIDATION_START, 50)
            restored_pools = restored_generator.generate(user_id, VALIDATION_START, 50)
            self.assertEqual(reference_pools, restored_pools)
            self.assertEqual(
                tuple(candidate.sources for candidate in reference_pools.candidates),
                tuple(candidate.sources for candidate in restored_pools.candidates),
            )
            reference_raw = tuple(
                ContextBuilder(reference.state).build(user_id, candidate, VALIDATION_START)
                for candidate in reference_pools.candidates
            )
            restored_raw = tuple(
                ContextBuilder(restored.state).build(user_id, candidate, VALIDATION_START)
                for candidate in restored_pools.candidates
            )
            self.assertEqual(reference_raw, restored_raw)
            self.assertEqual(
                reference.scaler.transform(reference_raw),
                restored.scaler.transform(restored_raw),
            )

    def test_sessionization_gap_threshold_is_strictly_greater_than_ten_minutes(self) -> None:
        times = [
            BASE,
            BASE + timedelta(minutes=5),
            BASE + timedelta(minutes=14),
            BASE + timedelta(minutes=24),
            BASE + timedelta(minutes=34, microseconds=1),
        ]
        starts = tuple(
            iter_session_starts(
                [event("U1", timestamp, f"T{index}") for index, timestamp in enumerate(times)]
            )
        )
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0].decision_time, times[0])
        self.assertEqual(starts[1].decision_time, times[4])

    def test_one_session_with_multiple_plays_yields_one_decision_point(self) -> None:
        events = [
            event("U1", BASE + timedelta(minutes=minute), f"T{index}")
            for index, minute in enumerate((0, 2, 5, 9, 10))
        ]
        starts = tuple(iter_session_starts(events))
        self.assertEqual(len(events), 5)
        self.assertEqual(len(starts), 1)

    def test_session_gap_uses_elapsed_time_across_daylight_saving(self) -> None:
        zone = ZoneInfo("America/New_York")
        before_jump = datetime(2024, 3, 10, 1, 55, tzinfo=zone)
        after_jump = datetime(2024, 3, 10, 3, 5, tzinfo=zone)
        starts = tuple(
            iter_session_starts(
                [
                    PlayEvent.from_raw("U1", before_jump, "T1", "A1"),
                    PlayEvent.from_raw("U1", after_jump, "T2", "A1"),
                ]
            )
        )
        self.assertEqual(len(starts), 1)

    def test_warmup_builds_state_and_scaler_without_learner_updates(self) -> None:
        simulator = build_simulator()
        initial_a, initial_b = simulator.learner.A, simulator.learner.b
        summary = prepare_default_warmup(simulator)
        self.assertEqual(summary.decision_points, 3)
        self.assertGreater(summary.context_rows, 0)
        self.assertTrue(simulator.scaler.is_fitted)
        self.assertEqual((simulator.learner.A, simulator.learner.b), (initial_a, initial_b))
        self.assertEqual(simulator.state.get_user_event_count("U1"), 3)

    def test_validation_decisions_are_chronological_and_boundary_is_invisible(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        result = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        self.assertEqual(
            [record.decision_time for record in result.decisions],
            [
                VALIDATION_START,
                VALIDATION_START + timedelta(days=1),
                VALIDATION_START + HORIZON,
            ],
        )
        first = result.decisions[0]
        self.assertNotIn(first.selected_track, {"V0", "V1", "V2"})
        self.assertEqual(first.selected_context[0], math.log1p(3))
        self.assertLessEqual(
            simulator.state.get_user_event_count("U1"), len(make_events())
        )

    def test_boundary_events_are_processed_after_the_simultaneous_decision_group(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        result = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        at_start = result.decisions[0]
        at_day_seven = result.decisions[2]
        self.assertNotIn("V0", {item.track_key for item in at_start.top_k_candidates})
        self.assertNotIn("V4", {item.track_key for item in at_day_seven.top_k_candidates})
        self.assertGreaterEqual(simulator.state.get_user_item_count("U1", resolve_item_key("V0", resolve_artist_key("A1", None), None)), 1)
        self.assertGreaterEqual(simulator.state.get_user_item_count("U1", resolve_item_key("V4", resolve_artist_key("A1", None), None)), 1)

    def test_simultaneous_users_share_the_same_strictly_prior_state(self) -> None:
        simulator = build_simulator(eligible_users={"U1", "U2"})
        prepare_default_warmup(simulator)
        result = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + HORIZON
        )
        same_time = [
            record
            for record in result.decisions
            if record.decision_time == VALIDATION_START
        ]
        self.assertEqual({record.user_id for record in same_time}, {"U1", "U2"})
        for record in same_time:
            candidate_keys = {item.track_key for item in record.top_k_candidates}
            self.assertNotIn("V0", candidate_keys)
            self.assertNotIn("U2V0", candidate_keys)
        u2_record = next(record for record in same_time if record.user_id == "U2")
        self.assertEqual(u2_record.selected_context[0], 0.0)

    def test_simultaneous_decisions_are_invariant_to_boundary_event_order(self) -> None:
        prefix = [
            event("U1", BASE, "T0"),
            event("U2", BASE, "T1", "A2"),
            event("U1", BASE + timedelta(minutes=20), "T2"),
            event("U2", BASE + timedelta(minutes=20), "T3", "A2"),
        ]
        boundary = [
            event("U1", VALIDATION_START, "U1_BOUNDARY", "A3"),
            event("U2", VALIDATION_START, "U2_BOUNDARY", "A4"),
        ]
        outputs = []
        for ordered_boundary in (boundary, list(reversed(boundary))):
            events = prefix + ordered_boundary + [
                event("U1", VALIDATION_START + HORIZON, "LATER")
            ]
            simulator = build_simulator(events, eligible_users={"U1", "U2"})
            simulator.prepare_warmup(BASE, BASE + timedelta(days=1))
            result = simulator.run_phase(
                "validation", VALIDATION_START, VALIDATION_START + HORIZON
            )
            simultaneous = {
                record.user_id: record
                for record in result.decisions
                if record.decision_time == VALIDATION_START
            }
            self.assertEqual(set(simultaneous), {"U1", "U2"})
            for record in simultaneous.values():
                keys = {candidate.track_key for candidate in record.top_k_candidates}
                self.assertNotIn("U1_BOUNDARY", keys)
                self.assertNotIn("U2_BOUNDARY", keys)
            outputs.append(simultaneous)

        for user_id in ("U1", "U2"):
            left, right = outputs[0][user_id], outputs[1][user_id]
            self.assertEqual(left.selected_context, right.selected_context)
            self.assertEqual(left.selected_track, right.selected_track)
            self.assertEqual(left.top_k_candidates, right.top_k_candidates)
            self.assertEqual(left.top_k_scores, right.top_k_scores)

    def test_delayed_feedback_is_scheduled_not_applied_at_selection(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        validation = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        first = validation.decisions[0]
        self.assertAlmostEqual(first.reward, math.log(5))
        self.assertEqual(first.reward_available_time, VALIDATION_START + HORIZON)
        self.assertEqual(first.feedback_applied_before, 0)
        self.assertEqual(validation.feedback_scheduled, 3)
        self.assertEqual(validation.feedback_applied, 3)
        self.assertEqual(validation.feedback_pending_at_end, 0)

    def test_intermediate_decision_does_not_receive_not_yet_available_feedback(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        validation = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        self.assertEqual(validation.decisions[1].decision_time, VALIDATION_START + timedelta(days=1))
        self.assertEqual(validation.decisions[1].feedback_applied_before, 0)

    def test_feedback_available_exactly_at_decision_updates_before_selection(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        validation = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        exact_boundary = validation.decisions[2]
        self.assertEqual(exact_boundary.decision_time, VALIDATION_START + HORIZON)
        self.assertEqual(exact_boundary.feedback_applied_before, 1)
        self.assertNotEqual(exact_boundary.linucb_mean, 0.0)

    def test_only_top1_context_is_scheduled_for_feedback(self) -> None:
        simulator = build_simulator(top_k=10)
        prepare_default_warmup(simulator)
        result = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        self.assertEqual(result.feedback_scheduled, sum(
            record.selected_candidate is not None for record in result.decisions
        ))
        self.assertTrue(
            all(record.selected_context is not None for record in result.decisions)
        )
        self.assertTrue(
            all(record.top_k_effective_size <= record.candidate_count for record in result.decisions)
        )

    def test_phase_boundary_excludes_reward_windows_that_cross_end(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        result = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=12)
        )
        self.assertEqual(
            [record.decision_time for record in result.decisions],
            [VALIDATION_START, VALIDATION_START + timedelta(days=1)],
        )

    def test_validation_and_test_reset_learner_but_keep_history_and_scaler(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        old_mean, old_scale = simulator.scaler.mean_, simulator.scaler.scale_
        validation = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        self.assertNotEqual(simulator.learner.b, (0.0,) * 15)
        validation_history_count = simulator.state.get_user_event_count("U1")

        test = simulator.run_phase("test", TEST_START, TEST_START + timedelta(days=14))
        self.assertEqual(simulator.scaler.mean_, old_mean)
        self.assertEqual(simulator.scaler.scale_, old_scale)
        self.assertGreater(validation_history_count, 3)
        self.assertEqual(test.decisions[0].phase, "test")
        self.assertEqual(test.decisions[0].linucb_mean, 0.0)
        self.assertEqual(test.decisions[0].selected_context[0], math.log1p(validation_history_count))
        self.assertNotEqual(validation.decisions[0].selected_context[0], test.decisions[0].selected_context[0])

    def test_ineligible_users_are_not_given_decision_records(self) -> None:
        simulator = build_simulator(eligible_users={"U1"})
        prepare_default_warmup(simulator)
        result = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        self.assertTrue(all(record.user_id == "U1" for record in result.decisions))

    def test_warmup_scaler_is_independent_of_eligibility_and_future_events(self) -> None:
        warmup_events = [
            event("U1", BASE, "T0", "A1"),
            event("U2", BASE, "T1", "A2"),
            event("U1", BASE + timedelta(minutes=20), "T2", "A1"),
            event("U2", BASE + timedelta(minutes=20), "T3", "A2"),
        ]
        baseline = build_simulator(warmup_events, eligible_users={"U1"})
        baseline_summary = baseline.prepare_warmup(
            BASE, BASE + timedelta(days=1)
        )

        with_future = warmup_events + [
            event("U9", BASE + timedelta(days=2), "FUTURE_ONLY", "A9")
        ]
        changed_eligibility = build_simulator(
            with_future, eligible_users={"U2"}
        )
        changed_summary = changed_eligibility.prepare_warmup(
            BASE, BASE + timedelta(days=1)
        )

        self.assertEqual(baseline_summary.decision_points, 4)
        self.assertEqual(changed_summary.decision_points, 4)
        self.assertEqual(baseline.scaler.mean_, changed_eligibility.scaler.mean_)
        self.assertEqual(baseline.scaler.scale_, changed_eligibility.scaler.scale_)
        self.assertEqual(baseline_summary.context_rows, changed_summary.context_rows)

    def test_prepare_warmup_rejects_scaler_without_auditable_provenance(self) -> None:
        prefit = FeatureScaler().fit(((0.0,) * 13 + (0.0, 0.0),))
        simulator = build_simulator(scaler=prefit)
        with self.assertRaisesRegex(RuntimeError, "unfitted FeatureScaler"):
            prepare_default_warmup(simulator)

    def test_empty_candidate_pool_is_logged_as_skipped(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        generate = simulator._candidate_generator.generate

        def empty_at_first_decision(user_id, timestamp, pool_depth=50):
            if timestamp == VALIDATION_START:
                return CandidatePools((), (), (), ())
            return generate(user_id, timestamp, pool_depth)

        with patch.object(
            simulator._candidate_generator,
            "generate",
            side_effect=empty_at_first_decision,
        ):
            result = simulator.run_phase(
                "validation", VALIDATION_START, VALIDATION_START + timedelta(days=15)
            )
        self.assertEqual(result.skipped_decisions, 1)
        skipped = result.decisions[0]
        self.assertEqual(skipped.candidate_count, 0)
        self.assertIsNone(skipped.selected_candidate)
        self.assertIsNone(skipped.reward)

    def test_top_k_returns_only_available_candidates_without_duplicates(self) -> None:
        simulator = build_simulator(top_k=100)
        prepare_default_warmup(simulator)
        result = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        for record in result.decisions:
            keys = [candidate.track_key for candidate in record.top_k_candidates]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(record.top_k_effective_size, min(100, record.candidate_count))

    def test_future_only_tracks_do_not_leak_into_candidates_or_context(self) -> None:
        simulator = build_simulator()
        prepare_default_warmup(simulator)
        result = simulator.run_phase(
            "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
        )
        first = result.decisions[0]
        all_candidate_keys = {item.track_key for item in first.top_k_candidates}
        self.assertNotIn("V0", all_candidate_keys)
        self.assertNotIn("V1", all_candidate_keys)
        self.assertEqual(first.selected_context[0], math.log1p(3))

    def test_identical_inputs_produce_identical_decision_logs(self) -> None:
        outputs = []
        for _ in range(2):
            simulator = build_simulator()
            prepare_default_warmup(simulator)
            outputs.append(
                simulator.run_phase(
                    "validation", VALIDATION_START, VALIDATION_START + timedelta(days=14)
                ).decisions
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_warmup_rejects_second_run_and_run_phase_requires_warmup(self) -> None:
        simulator = build_simulator()
        with self.assertRaisesRegex(RuntimeError, "prepare_warmup"):
            simulator.run_phase("validation", VALIDATION_START, VALIDATION_START + HORIZON)
        prepare_default_warmup(simulator)
        with self.assertRaisesRegex(RuntimeError, "already been prepared"):
            prepare_default_warmup(simulator)


if __name__ == "__main__":
    unittest.main()
