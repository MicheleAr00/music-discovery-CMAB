"""Experiment config, end-to-end smoke, fairness, and persistence tests."""

from datetime import date, datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from src.data.state import PlayEvent
from src.experiments.config import ExperimentConfig, TemporalProtocol
from src.experiments.dataset import load_lastfm_tsv
from src.experiments.runner import ExperimentRunner


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
        "candidate_pool_depths": (50,),
        "k_values": (5, 10, 20),
        "policies": ("random", "popularity", "greedy", "epsilon_greedy", "linucb"),
        "alpha_values": (0.0, 0.5),
        "epsilon_values": (0.0, 0.2, 1.0),
        "random_seed": 42,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


class ExperimentConfigTests(unittest.TestCase):
    def test_validation_config_file_loads_frozen_sweep_values(self):
        path = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "validation.json"
        config = ExperimentConfig.from_json(path)
        self.assertEqual(config.mode, "validation_sweep")
        self.assertEqual(config.n_train_unit, "events")
        self.assertEqual(config.candidate_pool_depths, (50, 100, 200))
        self.assertEqual(config.alpha_values, (0.0, 0.1, 0.25, 0.5, 1.0, 2.0))
        self.assertEqual(config.epsilon_values, (0.05, 0.1, 0.2))

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

    def test_parameter_sweeps_and_pool_depths_are_validated(self):
        with self.assertRaisesRegex(ValueError, "candidate_pool_depths"):
            validation_config(candidate_pool_depths=(75,))
        with self.assertRaisesRegex(ValueError, "k_values"):
            validation_config(k_values=(3,))


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

    def test_five_policy_smoke_uses_shared_schedule_and_candidate_pool(self):
        runner = self.make_runner()
        runs = runner.run_validation_sweep()
        self.assertEqual({run.policy for run in runs}, {
            "random", "popularity", "greedy", "epsilon_greedy", "linucb"
        })
        self.assertEqual(len(runs), 8)
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
        first = self.make_runner().run_validation_sweep()
        second = self.make_runner().run_validation_sweep()
        for left, right in zip(first, second):
            self.assertEqual(left.run_id, right.run_id)
            self.assertEqual(left.simulation.decisions, right.simulation.decisions)
            self.assertEqual(left.metrics, right.metrics)
            self.assertEqual(left.candidate_pool_digest, right.candidate_pool_digest)

    def test_validation_results_are_ranked_only_with_explicit_metric(self):
        runner = self.make_runner()
        runs = runner.run_validation_sweep()
        with self.assertRaisesRegex(ValueError, "selection metric"):
            runner.rank_validation_results(runs)
        ordered = runner.rank_validation_results(runs, metric="average_reward", direction="maximize")
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
            policies=tuple(params), alpha_values=(0.25,), epsilon_values=(0.1,),
        )
        runner = ExperimentRunner(tiny_events(), config)
        runs = runner.run_final_test()
        self.assertEqual(len(runs), 5)
        self.assertTrue(all(run.evaluation_phase == "test" for run in runs))
        self.assertEqual(len({run.candidate_pool_digest for run in runs}), 1)
        with self.assertRaisesRegex(RuntimeError, "already run"):
            runner.run_final_test()


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
