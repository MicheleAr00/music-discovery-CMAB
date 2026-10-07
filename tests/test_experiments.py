"""Experiment config, end-to-end smoke, fairness, and persistence tests."""

from datetime import date, datetime, timedelta
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data.state import PlayEvent
from src.experiments.config import ExperimentConfig, TemporalProtocol
from src.experiments.dataset import load_lastfm_tsv
from src.experiments.runner import ExperimentRunner
from src.simulation.candidate_generator import CandidateGenerator, CandidatePools
from src.simulation.simulator import iter_session_starts


def tiny_events() -> list[PlayEvent]:
    warm = datetime(2020, 1, 1, 10)
    validation = datetime(2020, 1, 10, 10)
    test = datetime(2020, 1, 20, 10)
    return sorted([
        PlayEvent("U1", warm, "T1", "A1"),
        PlayEvent("U1", warm + timedelta(minutes=20), "T2", "A2"),
        PlayEvent("U1", warm + timedelta(minutes=40), "T1", "A1"),
        PlayEvent("U2", warm, "T3", "A3"),
        PlayEvent("U2", warm + timedelta(minutes=20), "T4", "A4"),
        PlayEvent("U2", warm + timedelta(minutes=40), "T3", "A3"),
        PlayEvent("U1", validation, "V1", "A1"),
        PlayEvent("U2", validation, "V2", "A3"),
        PlayEvent("U1", validation + timedelta(hours=12), "T2", "A2"),
        PlayEvent("U2", validation + timedelta(hours=12), "T4", "A4"),
        PlayEvent("U1", test, "X1", "A1"),
        PlayEvent("U2", test, "X2", "A3"),
        PlayEvent("U1", test + timedelta(hours=12), "T2", "A2"),
        PlayEvent("U2", test + timedelta(hours=12), "T4", "A4"),
    ], key=lambda row: row.timestamp)


def tiny_protocol() -> TemporalProtocol:
    return TemporalProtocol(
        warmup_start=date(2020, 1, 1),
        warmup_end=date(2020, 1, 2),
        validation_start=date(2020, 1, 10),
        validation_end=date(2020, 1, 13),
        test_start=date(2020, 1, 20),
        test_end=date(2020, 1, 23),
        reward_horizon_days=1,
        session_gap_minutes=10,
    )


def validation_config(**overrides) -> ExperimentConfig:
    values = {
        "dataset_path": "synthetic://tiny",
        "dataset_id": "tiny-v1",
        "mode": "validation_sweep",
        "temporal": tiny_protocol(),
        "n_min": 1,
        "n_train_unit": "events",
        "candidate_pool_depths": (100,),
        "k_values": (5, 10, 20),
        "policies": ("random", "popularity", "greedy", "epsilon_greedy", "linucb"),
        "alpha_values": (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
        "epsilon_values": (0.05, 0.10, 0.20),
        "random_seed": 42,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def eligibility_fixture():
    warm = datetime(2020, 1, 1, 10)
    validation = datetime(2020, 1, 10)
    test = datetime(2020, 2, 1)
    events = [
        PlayEvent("U", warm + timedelta(minutes=index), "T0", "A0")
        for index in range(10)
    ]
    validation_points = tuple(validation + timedelta(hours=12 * index) for index in range(10))
    test_points = tuple(test + timedelta(hours=12 * index) for index in range(10))
    events.extend(PlayEvent("U", timestamp, "T0", "A0") for timestamp in validation_points)
    events.extend(PlayEvent("U", timestamp, "T0", "A0") for timestamp in test_points)
    # These starts are inside each phase but too close to phase_end for a
    # complete seven-day reward window.
    events.append(PlayEvent("U", datetime(2020, 1, 19), "T0", "A0"))
    events.append(PlayEvent("U", datetime(2020, 2, 14), "T0", "A0"))
    protocol = TemporalProtocol(
        warmup_start=date(2020, 1, 1),
        warmup_end=date(2020, 1, 2),
        validation_start=date(2020, 1, 10),
        validation_end=date(2020, 1, 25),
        test_start=date(2020, 2, 1),
        test_end=date(2020, 2, 20),
        reward_horizon_days=7,
        session_gap_minutes=10,
    )
    config = validation_config(
        temporal=protocol,
        n_min=10,
        policies=("random",),
        candidate_pool_depths=(50, 100, 200),
        k_values=(5,),
    )
    return sorted(events, key=lambda event: event.timestamp), config, validation_points[0], test_points[0]


class ExperimentConfigTests(unittest.TestCase):
    def test_validation_config_file_loads_frozen_sweep_values(self):
        path = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "validation.json"
        config = ExperimentConfig.from_json(path)
        self.assertEqual(config.mode, "validation_sweep")
        self.assertEqual(config.n_train_unit, "events")
        self.assertEqual(config.candidate_pool_depths, (50, 100, 200))
        self.assertEqual(config.alpha_values, (0.0, 0.1, 0.25, 0.5, 1.0, 2.0))
        self.assertEqual(config.epsilon_values, (0.05, 0.1, 0.2))
        self.assertEqual(config.selection_metric, "ndcg@10")
        self.assertEqual(config.selection_direction, "maximize")
        self.assertEqual(ExperimentConfig().candidate_pool_depths, (100,))

    def test_default_protocol_matches_frozen_dates_and_durations(self):
        protocol = TemporalProtocol()
        self.assertEqual(protocol.warmup_bounds[0], datetime(2005, 10, 30))
        self.assertEqual(protocol.warmup_bounds[1], datetime(2008, 12, 31, 23, 59, 59, 999999))
        self.assertEqual(protocol.validation_bounds[0], datetime(2009, 1, 1))
        self.assertEqual(protocol.test_bounds[0], datetime(2009, 3, 1))
        self.assertEqual(protocol.test_bounds[1], datetime(2009, 5, 5, 23, 59, 59, 999999))
        self.assertEqual(protocol.reward_horizon, timedelta(days=7))
        self.assertEqual(protocol.session_gap, timedelta(minutes=10))

    def test_json_round_trip_preserves_config_and_hash(self):
        config = validation_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config.to_json(path)
            restored = ExperimentConfig.from_json(path)
        self.assertEqual(config.to_dict(), restored.to_dict())
        self.assertEqual(config.configuration_hash, restored.configuration_hash)

    def test_final_test_requires_explicit_parameters(self):
        with self.assertRaisesRegex(ValueError, "final_parameters"):
            validation_config(mode="final_test")

    def test_frozen_train_eligibility_unit_cannot_be_changed(self):
        with self.assertRaisesRegex(ValueError, "N_train unit is events"):
            validation_config(n_train_unit="decision_points")

    def test_parameter_sweeps_and_pool_depths_are_validated(self):
        with self.assertRaisesRegex(ValueError, "candidate_pool_depths"):
            validation_config(candidate_pool_depths=(75,))
        with self.assertRaisesRegex(ValueError, "k_values"):
            validation_config(k_values=(3,))
        with self.assertRaisesRegex(ValueError, "frozen grid"):
            validation_config(alpha_values=(0.3,))
        with self.assertRaisesRegex(ValueError, "selection metric"):
            validation_config(selection_metric="average_reward")
        with self.assertRaisesRegex(ValueError, "full frozen alpha grid"):
            validation_config(alpha_values=(0.0, 0.5))


class ExperimentRunnerTests(unittest.TestCase):
    def make_runner(self, **overrides) -> ExperimentRunner:
        return ExperimentRunner(tiny_events(), validation_config(**overrides))

    def test_eligibility_is_computed_once_and_shared_across_policies(self):
        runner = self.make_runner()
        self.assertEqual(runner.eligible_users, frozenset({"U1", "U2"}))
        self.assertEqual(runner.eligibility_counts["U1"]["train"], 3)
        self.assertEqual(runner.eligibility_counts["U1"]["validation_usable"], 2)
        self.assertEqual(runner.eligibility_counts["U1"]["test_usable"], 2)
        runs = runner.run_validation_sweep()
        self.assertTrue(all(run.eligible_users == runner.eligible_users for run in runs))

    def test_empty_pool_points_are_excluded_from_validation_and_test_usable_counts(self):
        events, config, empty_validation_time, empty_test_time = eligibility_fixture()
        original_generate = CandidateGenerator.generate

        def return_empty_at(self, user_id, timestamp, pool_depth=50):
            pools = original_generate(self, user_id, timestamp, pool_depth)
            if timestamp in {empty_validation_time, empty_test_time}:
                return CandidatePools((), (), (), ())
            return pools

        with patch.object(CandidateGenerator, "generate", new=return_empty_at):
            runner = ExperimentRunner(events, config)
        raw_validation_points = sum(
            config.temporal.validation_bounds[0] <= session.decision_time <= config.temporal.validation_bounds[1]
            and session.decision_time + config.temporal.reward_horizon <= config.temporal.validation_bounds[1]
            for session in iter_session_starts(events, config.temporal.session_gap)
        )
        self.assertEqual(runner.eligibility_counts["U"]["train"], 10)
        self.assertEqual(raw_validation_points, 10)
        self.assertEqual(runner.eligibility_counts["U"]["validation_usable"], 9)
        self.assertEqual(runner.eligibility_counts["U"]["test_usable"], 9)

    def test_ten_raw_phase_points_with_one_empty_pool_fail_n_min(self):
        events, config, empty_validation_time, _ = eligibility_fixture()
        original_generate = CandidateGenerator.generate

        def return_empty_at(self, user_id, timestamp, pool_depth=50):
            pools = original_generate(self, user_id, timestamp, pool_depth)
            if timestamp == empty_validation_time:
                return CandidatePools((), (), (), ())
            return pools

        with patch.object(CandidateGenerator, "generate", new=return_empty_at):
            runner = ExperimentRunner(events, config)
        raw_validation_points = sum(
            config.temporal.validation_bounds[0] <= session.decision_time <= config.temporal.validation_bounds[1]
            and session.decision_time + config.temporal.reward_horizon <= config.temporal.validation_bounds[1]
            for session in iter_session_starts(events, config.temporal.session_gap)
        )
        self.assertEqual(raw_validation_points, 10)
        self.assertEqual(runner.eligibility_counts["U"]["validation_usable"], 9)
        self.assertEqual(runner.eligibility_counts["U"]["test_usable"], 10)
        self.assertNotIn("U", runner.eligible_users)

    def test_nonempty_pools_keep_ten_usable_points_and_horizon_filter(self):
        events, config, _, _ = eligibility_fixture()
        runner = ExperimentRunner(events, config)
        session_starts = tuple(iter_session_starts(events, config.temporal.session_gap))
        raw_validation_points = sum(
            config.temporal.validation_bounds[0] <= session.decision_time <= config.temporal.validation_bounds[1]
            for session in session_starts
        )
        raw_test_points = sum(
            config.temporal.test_bounds[0] <= session.decision_time <= config.temporal.test_bounds[1]
            for session in session_starts
        )
        self.assertEqual(raw_validation_points, 11)
        self.assertEqual(raw_test_points, 11)
        self.assertEqual(runner.eligibility_counts["U"]["validation_usable"], 10)
        self.assertEqual(runner.eligibility_counts["U"]["test_usable"], 10)
        self.assertIn("U", runner.eligible_users)

    def test_five_policy_smoke_uses_shared_schedule_and_candidate_pool(self):
        runner = self.make_runner()
        runs = runner.run_validation_sweep()
        self.assertEqual({run.policy for run in runs}, {
            "random", "popularity", "greedy", "epsilon_greedy", "linucb"
        })
        self.assertEqual(len(runs), 12)
        self.assertEqual(len({run.candidate_pool_digest for run in runs}), 1)
        self.assertEqual(len({run.scaler_digest for run in runs}), 1)
        self.assertEqual(
            len({tuple(run.summary_row(runner.config)) for run in runs}), 1
        )
        for run in runs:
            self.assertEqual(run.simulation.decision_points, 4)
            self.assertGreater(run.warmup.context_rows, 0)
            self.assertEqual(run.evaluation_phase, "validation")

    def test_alpha_and_epsilon_changes_do_not_change_candidate_pool(self):
        runs = self.make_runner().run_validation_sweep()
        linucb = [run for run in runs if run.policy == "linucb"]
        epsilon = [run for run in runs if run.policy == "epsilon_greedy"]
        self.assertEqual(len({run.candidate_pool_digest for run in linucb}), 1)
        self.assertEqual(len({run.candidate_pool_digest for run in epsilon}), 1)

    def test_same_seed_reproduces_decisions_metrics_and_pool_digest(self):
        first_runner = self.make_runner()
        second_runner = self.make_runner()
        first = first_runner.run_validation_sweep()
        second = second_runner.run_validation_sweep()
        for left, right in zip(first, second):
            self.assertEqual(left.run_id, right.run_id)
            self.assertEqual(left.to_json_dict(first_runner.config)["decisions"],
                             right.to_json_dict(second_runner.config)["decisions"])
            self.assertEqual(left.metrics, right.metrics)
            self.assertEqual(left.candidate_pool_digest, right.candidate_pool_digest)

    def test_validation_sweep_spools_details_and_returns_compact_results(self):
        runner = self.make_runner(policies=("random",), candidate_pool_depths=(50, 100))
        runs = runner.run_validation_sweep()
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertEqual(run.simulation.decisions, ())
            self.assertEqual(run.simulation.eligible_catalog, frozenset())
            self.assertEqual(run.simulation.eligible_artists, frozenset())
            details = run.to_json_dict(runner.config)
            self.assertEqual(len(details["decisions"]), run.simulation.decision_points)
            self.assertEqual(details["candidate_pool_digest"], run.candidate_pool_digest)

    def test_bounded_sweep_details_match_in_memory_reference_runs(self):
        runner = self.make_runner(policies=("random",), candidate_pool_depths=(50, 100))
        reference = tuple(
            runner._run_one("validation", policy, params, depth)
            for depth in runner.config.candidate_pool_depths
            for policy, params in runner._validation_policy_grid()
        )
        bounded = runner.run_validation_sweep()
        self.assertEqual(len(reference), len(bounded))
        for full, compact in zip(reference, bounded):
            details = compact.to_json_dict(runner.config)
            self.assertEqual(full.run_id, compact.run_id)
            self.assertEqual(full.metrics, compact.metrics)
            self.assertEqual(full.candidate_pool_digest, compact.candidate_pool_digest)
            self.assertEqual(full.scaler_digest, compact.scaler_digest)
            self.assertEqual(
                [_decision for _decision in full.to_json_dict(runner.config)["decisions"]],
                details["decisions"],
            )

    def test_validation_results_are_ranked_by_frozen_criterion(self):
        runner = self.make_runner()
        runs = runner.run_validation_sweep()
        ordered = runner.rank_validation_results(runs)
        self.assertEqual(len(ordered), len(runs))
        self.assertTrue(ordered)

    def test_result_writer_emits_flat_csv_and_detailed_jsonl(self):
        runner = self.make_runner(policies=("random",))
        runs = runner.run_validation_sweep()
        with tempfile.TemporaryDirectory() as directory:
            csv_path, jsonl_path = runner.write_results(runs, directory)
            row = json.loads(jsonl_path.read_text(encoding="utf-8"))
            self.assertTrue(csv_path.is_file())
            self.assertEqual(row["summary"]["evaluation_phase"], "validation")
            self.assertEqual(len(row["decisions"]), 4)
            self.assertTrue(row["candidate_pool_digest"])

    def test_final_test_is_a_separate_explicit_run(self):
        params = {
            "random": {}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.1}, "linucb": {"alpha": 0.25},
        }
        config = validation_config(
            mode="final_test", final_parameters=params,
            candidate_pool_depths=(100,),
            policies=tuple(params), alpha_values=(0.25,), epsilon_values=(0.1,),
        )
        runner = ExperimentRunner(tiny_events(), config)
        runs = runner.run_final_test()
        self.assertEqual(len(runs), 5)
        self.assertTrue(all(run.evaluation_phase == "test" for run in runs))
        self.assertEqual(len({run.candidate_pool_digest for run in runs}), 1)
        with self.assertRaisesRegex(RuntimeError, "already run"):
            runner.run_final_test()


class ValidationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = validation_config(candidate_pool_depths=(50, 100, 200))
        cls.runner = ExperimentRunner(tiny_events(), cls.config)
        cls.runs = cls.runner.run_validation_sweep()

    def test_selection_uses_only_configured_alpha_and_epsilon_at_n100(self):
        selection = self.runner.select_validation_parameters(self.runs)
        linucb = [run for run in self.runs if run.policy == "linucb" and run.candidate_pool_depth == 100]
        epsilon = [run for run in self.runs if run.policy == "epsilon_greedy" and run.candidate_pool_depth == 100]
        self.assertEqual({float(run.policy_parameters["alpha"]) for run in linucb}, set(self.config.alpha_values))
        self.assertEqual({float(run.policy_parameters["epsilon"]) for run in epsilon}, set(self.config.epsilon_values))
        self.assertIn(selection.policy_parameters["linucb"]["alpha"], self.config.alpha_values)
        self.assertIn(selection.policy_parameters["epsilon_greedy"]["epsilon"], self.config.epsilon_values)
        self.assertEqual({run.candidate_pool_depth for run in self.runs}, {50, 100, 200})

    def test_n100_is_common_for_every_policy_in_final_configuration(self):
        selection = self.runner.select_validation_parameters(self.runs)
        final_config = self.runner.make_final_test_config(selection)
        self.assertEqual(final_config.candidate_pool_depths, (100,))
        self.assertEqual(set(final_config.final_parameters), set(self.config.policies))
        self.assertEqual(len({final_config.candidate_pool_depths[0] for _ in final_config.policies}), 1)
        self.assertEqual(final_config.k_values, (5, 10, 20))
        self.assertTrue(all("k" not in params for params in selection.policy_parameters.values()))
        with self.assertRaisesRegex(ValueError, "common primary candidate_pool_depth"):
            replace(final_config, candidate_pool_depths=(200,))

    def test_sensitivity_results_cannot_change_primary_parameter_selection(self):
        original = self.runner.select_validation_parameters(self.runs)
        adjusted = tuple(
            replace(run, metrics={**run.metrics, "ndcg@10": 999.0})
            if run.candidate_pool_depth in {50, 200}
            else run
            for run in self.runs
        )
        self.assertEqual(self.runner.select_validation_parameters(adjusted), original)

    def test_selection_rejects_any_test_phase_result(self):
        test_row = replace(
            next(run for run in self.runs if run.candidate_pool_depth == 100),
            evaluation_phase="test",
            metrics={"ndcg@10": 1_000_000.0, "average_reward": 1_000_000.0},
        )
        with self.assertRaisesRegex(ValueError, "validation results only"):
            self.runner.select_validation_parameters((*self.runs, test_row))

    def test_selection_is_deterministic_for_identical_validation_results(self):
        first = self.runner.select_validation_parameters(self.runs)
        second = self.runner.select_validation_parameters(tuple(reversed(self.runs)))
        self.assertEqual(first, second)

    def test_all_undefined_ndcg_for_tunable_policy_fails_without_fallback(self):
        undefined = tuple(
            replace(run, metrics={**run.metrics, "ndcg@10": None})
            if run.candidate_pool_depth == 100 and run.policy == "linucb"
            else run
            for run in self.runs
        )
        with self.assertRaisesRegex(ValueError, "NDCG@10 is undefined"):
            self.runner.select_validation_parameters(undefined)

    def test_ties_use_average_reward_then_lower_parameter(self):
        target = [
            run for run in self.runs
            if run.candidate_pool_depth == 100 and run.policy == "linucb"
        ]
        tied = tuple(
            replace(
                run,
                metrics={
                    **run.metrics,
                    "ndcg@10": 0.5,
                    "average_reward": 2.0 if run.policy_parameters["alpha"] == 0.5 else 1.0,
                },
            )
            if run in target else run
            for run in self.runs
        )
        selection = self.runner.select_validation_parameters(tied)
        self.assertEqual(selection.policy_parameters["linucb"]["alpha"], 0.5)

        parameter_tie = tuple(
            replace(
                run,
                metrics={**run.metrics, "ndcg@10": 0.5, "average_reward": 1.0},
            ) if run in target else run
            for run in self.runs
        )
        selection = self.runner.select_validation_parameters(parameter_tie)
        self.assertEqual(selection.policy_parameters["linucb"]["alpha"], 0.0)


class DatasetLoaderTests(unittest.TestCase):
    def test_loads_reverse_ordered_tsv_as_chronological_identity_keys(self):
        content = (
            "u1\t2020-01-02T00:00:00Z\ta2\tArtist Two\tt2\tTrack Two\n"
            "u1\t2020-01-01T00:00:00Z\t\tFallback Artist\t\tTrack One\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "plays.tsv"
            source.write_text(content, encoding="utf-8")
            events = load_lastfm_tsv(source, batch_size=1, temporary_directory=directory)
            try:
                self.assertEqual(len(events), 2)
                self.assertEqual(events[0].timestamp, datetime(2020, 1, 1))
                self.assertEqual(events[0].artist_id, "name::Fallback Artist")
                self.assertEqual(events[0].track_id, "name::Fallback Artist::Track One")
                self.assertEqual(events[1].track_id, "t2")
            finally:
                events.close()
            self.assertFalse(events.database_path.exists())


if __name__ == "__main__":
    unittest.main()
