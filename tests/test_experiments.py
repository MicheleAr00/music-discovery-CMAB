"""Experiment config, end-to-end smoke, fairness, and persistence tests."""

from datetime import date, datetime, timedelta
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

from src.data.state import PlayEvent
from src.data.identity import resolve_artist_key, resolve_event_identity_keys
from src.experiments.cli import _workspace_cache_directory
from src.experiments.config import ExperimentConfig, TemporalProtocol
from src.experiments.dataset import load_lastfm_tsv
import src.experiments.runner as runner_module
from src.experiments.runner import CheckpointError, ExperimentRunner, _experiment_code_digest
from src.simulation.candidate_generator import CandidateGenerator, CandidatePools
from src.simulation.scaler import FeatureScaler
from src.simulation.simulator import iter_session_starts


def tiny_events() -> list[PlayEvent]:
    warm = datetime(2020, 1, 1, 10)
    validation = datetime(2020, 1, 10, 10)
    test = datetime(2020, 1, 20, 10)
    return sorted([
        PlayEvent.from_raw("U1", warm, "T1", "A1"),
        PlayEvent.from_raw("U1", warm + timedelta(minutes=20), "T2", "A2"),
        PlayEvent.from_raw("U1", warm + timedelta(minutes=40), "T1", "A1"),
        PlayEvent.from_raw("U2", warm, "T3", "A3"),
        PlayEvent.from_raw("U2", warm + timedelta(minutes=20), "T4", "A4"),
        PlayEvent.from_raw("U2", warm + timedelta(minutes=40), "T3", "A3"),
        PlayEvent.from_raw("U1", validation, "V1", "A1"),
        PlayEvent.from_raw("U2", validation, "V2", "A3"),
        PlayEvent.from_raw("U1", validation + timedelta(hours=12), "T2", "A2"),
        PlayEvent.from_raw("U2", validation + timedelta(hours=12), "T4", "A4"),
        PlayEvent.from_raw("U1", test, "X1", "A1"),
        PlayEvent.from_raw("U2", test, "X2", "A3"),
        PlayEvent.from_raw("U1", test + timedelta(hours=12), "T2", "A2"),
        PlayEvent.from_raw("U2", test + timedelta(hours=12), "T4", "A4"),
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
        PlayEvent.from_raw("U", warm + timedelta(minutes=index), "T0", "A0")
        for index in range(10)
    ]
    validation_points = tuple(validation + timedelta(hours=12 * index) for index in range(10))
    test_points = tuple(test + timedelta(hours=12 * index) for index in range(10))
    events.extend(PlayEvent.from_raw("U", timestamp, "T0", "A0") for timestamp in validation_points)
    events.extend(PlayEvent.from_raw("U", timestamp, "T0", "A0") for timestamp in test_points)
    # These starts are inside each phase but too close to phase_end for a
    # complete seven-day reward window.
    events.append(PlayEvent.from_raw("U", datetime(2020, 1, 19), "T0", "A0"))
    events.append(PlayEvent.from_raw("U", datetime(2020, 2, 14), "T0", "A0"))
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
    def setUp(self):
        self._temporary_root = tempfile.TemporaryDirectory()
        self._runner_index = 0

    def tearDown(self):
        self._temporary_root.cleanup()

    def make_runner(self, **overrides) -> ExperimentRunner:
        index = self._runner_index
        self._runner_index += 1
        config = validation_config(
            results_directory=str(Path(self._temporary_root.name) / "results"),
            **overrides,
        )
        runner = ExperimentRunner(tiny_events(), config)
        runner._test_checkpoint_directory = Path(self._temporary_root.name) / f"checkpoints-{index}"
        return runner

    def run_sweep(self, runner, checkpoint_directory=None):
        directory = checkpoint_directory or runner._test_checkpoint_directory
        return runner.run_validation_sweep(directory)

    def test_eligibility_is_computed_once_and_shared_across_policies(self):
        runner = self.make_runner()
        self.assertEqual(runner.eligible_users, frozenset({"U1", "U2"}))
        self.assertEqual(runner.eligibility_counts["U1"]["train"], 3)
        self.assertEqual(runner.eligibility_counts["U1"]["validation_usable"], 2)
        self.assertEqual(runner.eligibility_counts["U1"]["test_usable"], 2)
        runs = self.run_sweep(runner)
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
        runs = self.run_sweep(runner)
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
        runner = self.make_runner()
        runs = self.run_sweep(runner)
        linucb = [run for run in runs if run.policy == "linucb"]
        epsilon = [run for run in runs if run.policy == "epsilon_greedy"]
        self.assertEqual(len({run.candidate_pool_digest for run in linucb}), 1)
        self.assertEqual(len({run.candidate_pool_digest for run in epsilon}), 1)

    def test_validation_fits_scaler_exactly_once_per_candidate_depth(self):
        runner = self.make_runner(
            candidate_pool_depths=(50, 100, 200),
        )
        original_fit = FeatureScaler.fit
        fitted_instances = []

        def counted_fit(scaler, contexts):
            fitted_instances.append(scaler)
            return original_fit(scaler, contexts)

        with patch.object(FeatureScaler, "fit", new=counted_fit):
            runs = self.run_sweep(runner)

        self.assertEqual(len(runs), 36)
        self.assertEqual(len(fitted_instances), 3)
        self.assertEqual(set(runner._warmup_artifacts), {50, 100, 200})
        self.assertEqual(len({id(scaler) for scaler in fitted_instances}), 3)
        for depth in (50, 100, 200):
            depth_runs = [run for run in runs if run.candidate_pool_depth == depth]
            self.assertEqual(len({run.scaler_digest for run in depth_runs}), 1)

    def test_cached_warmup_matches_reference_for_every_policy(self):
        runner = self.make_runner(
            candidate_pool_depths=(50, 100),
            policies=("random", "popularity", "greedy", "epsilon_greedy", "linucb"),
        )
        for policy_name, params in runner._validation_policy_grid():
            with self.subTest(policy=policy_name, parameters=params):
                reference = runner._run_one(
                    "validation", policy_name, params, 50
                )
                artifact = runner._warmup_artifacts[50]
                optimized = runner._run_one(
                    "validation", policy_name, params, 50,
                    warmup_artifact=artifact,
                )

                self.assertEqual(reference.warmup, optimized.warmup)
                self.assertEqual(reference.scaler_digest, optimized.scaler_digest)
                self.assertEqual(reference.candidate_pool_digest, optimized.candidate_pool_digest)
                self.assertEqual(reference.simulation.decisions, optimized.simulation.decisions)
                self.assertEqual(reference.simulation.feedback_scheduled, optimized.simulation.feedback_scheduled)
                self.assertEqual(reference.simulation.feedback_applied, optimized.simulation.feedback_applied)
                self.assertEqual(reference.simulation.feedback_pending_at_end, optimized.simulation.feedback_pending_at_end)
                self.assertEqual(reference.metrics, optimized.metrics)

    def test_warmup_artifact_cannot_be_reused_at_another_depth(self):
        runner = self.make_runner(candidate_pool_depths=(50, 100))
        runner._run_one("validation", "random", {}, 50)
        artifact = runner._warmup_artifacts[50]
        with self.assertRaisesRegex(ValueError, "provenance"):
            runner._run_one(
                "validation", "random", {}, 100,
                warmup_artifact=artifact,
            )

    def test_same_seed_reproduces_decisions_metrics_and_pool_digest(self):
        first_runner = self.make_runner()
        second_runner = self.make_runner()
        first = self.run_sweep(first_runner)
        second = self.run_sweep(second_runner)
        for left, right in zip(first, second):
            self.assertEqual(left.run_id, right.run_id)
            self.assertEqual(left.to_json_dict(first_runner.config)["decisions"],
                             right.to_json_dict(second_runner.config)["decisions"])
            self.assertEqual(left.metrics, right.metrics)
            self.assertEqual(left.candidate_pool_digest, right.candidate_pool_digest)

    def test_validation_sweep_spools_details_and_returns_compact_results(self):
        runner = self.make_runner(policies=("random",), candidate_pool_depths=(50, 100))
        runs = self.run_sweep(runner)
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
        bounded = self.run_sweep(runner)
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
        runs = self.run_sweep(runner)
        ordered = runner.rank_validation_results(runs)
        self.assertEqual(len(ordered), len(runs))
        self.assertTrue(ordered)

    def test_result_writer_emits_flat_csv_and_detailed_jsonl(self):
        runner = self.make_runner(policies=("random",))
        runs = self.run_sweep(runner)
        with tempfile.TemporaryDirectory() as directory:
            csv_path, jsonl_path = runner.write_results(runs, directory)
            row = json.loads(jsonl_path.read_text(encoding="utf-8"))
            self.assertTrue(csv_path.is_file())
            self.assertEqual(row["summary"]["evaluation_phase"], "validation")
            self.assertEqual(len(row["decisions"]), 4)
            self.assertTrue(row["candidate_pool_digest"])

    def test_checkpoint_resume_matches_uninterrupted_sweep_exactly(self):
        interrupted = self.make_runner()
        checkpoint_directory = interrupted._test_checkpoint_directory
        original_run_one = interrupted._run_one
        calls = 0

        def fail_after_one(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return original_run_one(*args, **kwargs)

        with patch.object(interrupted, "_run_one", side_effect=fail_after_one):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_sweep(interrupted)
        self.assertEqual(len(tuple(checkpoint_directory.glob("*.checkpoint.json"))), 1)

        resumed = self.make_runner()
        original_fit = FeatureScaler.fit
        fit_calls = []

        def counted_fit(scaler, contexts):
            fit_calls.append(scaler)
            return original_fit(scaler, contexts)

        with patch.object(FeatureScaler, "fit", new=counted_fit):
            resumed_runs = self.run_sweep(resumed, checkpoint_directory)
        reference = self.make_runner()
        reference_runs = self.run_sweep(reference)

        self.assertEqual(resumed.resumed_run_count, 1)
        # One N-specific warm-up is reconstructed on restart, then reused by
        # all remaining configurations at that depth.
        self.assertEqual(len(fit_calls), 1)
        self.assertEqual(len(resumed_runs), 12)
        self.assertEqual(len({run.run_id for run in resumed_runs}), 12)
        self.assertEqual(
            [run.run_id for run in resumed_runs],
            [run.run_id for run in reference_runs],
        )
        for resumed_run, reference_run in zip(resumed_runs, reference_runs):
            self.assertEqual(resumed_run.metrics, reference_run.metrics)
            self.assertEqual(resumed_run.candidate_pool_digest, reference_run.candidate_pool_digest)
            self.assertEqual(resumed_run.scaler_digest, reference_run.scaler_digest)
            self.assertEqual(
                resumed_run.to_json_dict(resumed.config)["decisions"],
                reference_run.to_json_dict(reference.config)["decisions"],
            )
        csv_path, jsonl_path = resumed.write_results(
            resumed_runs, Path(self._temporary_root.name) / "aggregated"
        )
        self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 13)
        self.assertEqual(len(jsonl_path.read_text(encoding="utf-8").splitlines()), 12)

    def _make_single_checkpoint(self):
        runner = self.make_runner(policies=("random",), candidate_pool_depths=(100,))
        runs = self.run_sweep(runner)
        self.assertEqual(len(runs), 1)
        checkpoint = next(runner._test_checkpoint_directory.glob("*.checkpoint.json"))
        detail = runner._test_checkpoint_directory / checkpoint.name.removesuffix(".checkpoint.json")
        detail = detail.with_suffix(".jsonl")
        return runner, checkpoint, detail

    def test_incomplete_checkpoint_is_not_treated_as_completed(self):
        runner, marker_path, _ = self._make_single_checkpoint()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["completion_status"] = "writing"
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        resumed = self.make_runner(policies=("random",), candidate_pool_depths=(100,))
        with self.assertRaisesRegex(CheckpointError, "not marked complete"):
            self.run_sweep(resumed, runner._test_checkpoint_directory)

    def test_corrupt_checkpoint_detail_is_not_reused(self):
        runner, _, detail_path = self._make_single_checkpoint()
        with detail_path.open("ab") as stream:
            stream.write(b"corruption")
        resumed = self.make_runner(policies=("random",), candidate_pool_depths=(100,))
        with self.assertRaisesRegex(CheckpointError, "checksum mismatch"):
            self.run_sweep(resumed, runner._test_checkpoint_directory)

    def test_incompatible_checkpoint_is_not_reused(self):
        runner, marker_path, _ = self._make_single_checkpoint()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["identity"]["dataset_content_hash"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        resumed = self.make_runner(policies=("random",), candidate_pool_depths=(100,))
        with self.assertRaisesRegex(CheckpointError, "dataset_content_hash differs"):
            self.run_sweep(resumed, runner._test_checkpoint_directory)

    def test_marker_with_wrong_checksum_is_rejected(self):
        runner, marker_path, _ = self._make_single_checkpoint()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["result_sha256"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        resumed = self.make_runner(policies=("random",), candidate_pool_depths=(100,))
        with self.assertRaisesRegex(CheckpointError, "checksum mismatch"):
            self.run_sweep(resumed, runner._test_checkpoint_directory)

    def test_partial_jsonl_without_completion_marker_is_rejected(self):
        runner = self.make_runner(policies=("random",))
        checkpoint_directory = runner._test_checkpoint_directory
        checkpoint_directory.mkdir(parents=True)
        (checkpoint_directory / "unfinished.jsonl").write_text(
            '{"summary":', encoding="utf-8"
        )
        resumed = self.make_runner(policies=("random",))
        with self.assertRaisesRegex(CheckpointError, "no completion marker"):
            self.run_sweep(resumed, checkpoint_directory)

    def test_checkpoint_from_different_code_digest_is_rejected(self):
        runner, marker_path, _ = self._make_single_checkpoint()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["identity"]["experiment_code_digest"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        resumed = self.make_runner(policies=("random",), candidate_pool_depths=(100,))
        with self.assertRaisesRegex(CheckpointError, "experiment_code_digest differs"):
            self.run_sweep(resumed, runner._test_checkpoint_directory)

    def test_checkpoint_from_different_configuration_is_rejected(self):
        runner, marker_path, _ = self._make_single_checkpoint()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["identity"]["configuration_hash"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        resumed = self.make_runner(policies=("random",), candidate_pool_depths=(100,))
        with self.assertRaisesRegex(CheckpointError, "configuration_hash differs"):
            self.run_sweep(resumed, runner._test_checkpoint_directory)

    def test_completion_marker_follows_closed_detail_and_checksum(self):
        runner = self.make_runner(policies=("random",))
        events = []
        detail_streams = []
        original_atomic_write = runner_module._atomic_write
        original_sha256_file = runner_module._sha256_file

        def tracked_atomic_write(path, writer):
            is_marker = path.name.endswith(".checkpoint.json")
            if is_marker:
                self.assertTrue(detail_streams[-1].closed)
                self.assertTrue(path.with_name(path.name.removesuffix(".checkpoint.json") + ".jsonl").is_file())
                events.append("marker")

            def capture_stream(stream):
                writer(stream)
                if not is_marker:
                    detail_streams.append(stream)

            original_atomic_write(path, capture_stream)
            if not is_marker:
                events.append("detail-closed")

        def tracked_sha256_file(path):
            events.append("checksum")
            return original_sha256_file(path)

        with patch.object(runner_module, "_atomic_write", side_effect=tracked_atomic_write), patch.object(
            runner_module, "_sha256_file", side_effect=tracked_sha256_file
        ):
            runs = self.run_sweep(runner)
        self.assertEqual(len(runs), 1)
        self.assertEqual(events[:3], ["detail-closed", "checksum", "marker"])

    def test_checkpoint_resume_across_independent_python_processes(self):
        root = Path(self._temporary_root.name)
        checkpoint_directory = root / "process-checkpoints"
        resumed_output = root / "resumed-signature.json"
        reference_output = root / "reference-signature.json"
        worker = r'''import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_experiments import tiny_events, validation_config
from src.experiments.runner import ExperimentRunner
mode, checkpoint, output, root = sys.argv[1:]
config = validation_config(results_directory=str(Path(root) / "results"))
runner = ExperimentRunner(tiny_events(), config)
if mode == "interrupt":
    original = runner._run_one
    calls = 0
    def fail_after_one(*args, **kwargs):
        global calls
        calls += 1
        if calls == 2:
            raise RuntimeError("intentional process boundary")
        return original(*args, **kwargs)
    runner._run_one = fail_after_one
    try:
        runner.run_validation_sweep(checkpoint)
    except RuntimeError as error:
        if str(error) != "intentional process boundary":
            raise
    else:
        raise AssertionError("expected the first process to stop after one run")
    if len(list(Path(checkpoint).glob("*.checkpoint.json"))) != 1:
        raise AssertionError("first process did not persist exactly one checkpoint")
elif mode in {"resume", "reference"}:
    runs = runner.run_validation_sweep(checkpoint)
    if mode == "resume" and runner.resumed_run_count != 1:
        raise AssertionError("second process did not resume exactly one completed run")
    signature = [
        {
            "run_id": run.run_id,
            "metrics": dict(run.metrics),
            "candidate_pool_digest": run.candidate_pool_digest,
            "scaler_digest": run.scaler_digest,
            "decisions": run.to_json_dict(config)["decisions"],
        }
        for run in runs
    ]
    Path(output).write_text(json.dumps(signature, sort_keys=True), encoding="utf-8")
else:
    raise AssertionError(mode)
'''

        def launch(mode, checkpoint, output=None):
            args = [sys.executable, "-c", worker, mode, str(checkpoint), str(output or ""), str(root)]
            return subprocess.run(
                args, cwd=Path.cwd(), capture_output=True, text=True, check=True
            )

        launch("interrupt", checkpoint_directory)
        marker_path = next(checkpoint_directory.glob("*.checkpoint.json"))
        detail_path = checkpoint_directory / marker_path.name.removesuffix(".checkpoint.json")
        detail_path = detail_path.with_suffix(".jsonl")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        import hashlib
        self.assertEqual(
            hashlib.sha256(detail_path.read_bytes()).hexdigest(),
            marker["result_sha256"],
        )
        launch("resume", checkpoint_directory, resumed_output)
        launch("reference", root / "reference-checkpoints", reference_output)
        self.assertEqual(
            json.loads(resumed_output.read_text(encoding="utf-8")),
            json.loads(reference_output.read_text(encoding="utf-8")),
        )

    def test_final_test_does_not_read_or_write_validation_checkpoints(self):
        params = {
            "random": {}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.1}, "linucb": {"alpha": 0.25},
        }
        config = validation_config(
            mode="final_test", final_parameters=params,
            candidate_pool_depths=(100,), policies=tuple(params),
            alpha_values=(0.25,), epsilon_values=(0.1,),
            results_directory=str(Path(self._temporary_root.name) / "final-results"),
        )
        runner = ExperimentRunner(tiny_events(), config)
        checkpoint_directory = Path(self._temporary_root.name) / "final-checkpoints"
        runs = runner.run_final_test(checkpoint_directory)
        self.assertEqual(len(runs), 5)
        self.assertEqual(runner.checkpoint_directory, checkpoint_directory.resolve())
        markers = tuple(checkpoint_directory.glob("*.checkpoint.json"))
        self.assertEqual(len(markers), 5)
        for marker in markers:
            record = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(record["identity"]["evaluation_phase"], "test")
            detail = checkpoint_directory / record["result_filename"]
            self.assertEqual(
                hashlib.sha256(detail.read_bytes()).hexdigest(), record["result_sha256"]
            )
        self.assertEqual(len(tuple(checkpoint_directory.glob("warmup-N100.json"))), 1)
        self.assertFalse(tuple(Path(self._temporary_root.name).glob("validation-checkpoints/*.checkpoint.json")))

    def test_final_test_is_a_separate_explicit_run(self):
        params = {
            "random": {}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.1}, "linucb": {"alpha": 0.25},
        }
        config = validation_config(
            mode="final_test", final_parameters=params,
            candidate_pool_depths=(100,),
            policies=tuple(params), alpha_values=(0.25,), epsilon_values=(0.1,),
            results_directory=str(Path(self._temporary_root.name) / "explicit-final-results"),
        )
        runner = ExperimentRunner(tiny_events(), config)
        runs = runner.run_final_test()
        self.assertEqual(len(runs), 5)
        self.assertTrue(all(run.evaluation_phase == "test" for run in runs))
        self.assertEqual(len({run.candidate_pool_digest for run in runs}), 1)
        with self.assertRaisesRegex(RuntimeError, "already run"):
            runner.run_final_test()

    def test_final_test_warmup_reuse_is_equivalent_to_original_path(self):
        params = {
            "random": {"random_seed": 42}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.05, "random_seed": 42},
            "linucb": {"alpha": 2.0},
        }
        config = validation_config(
            mode="final_test", final_parameters=params, candidate_pool_depths=(100,),
            results_directory=str(Path(self._temporary_root.name) / "final-results"),
        )
        reference_runner = ExperimentRunner(tiny_events(), config)
        reference = tuple(
            reference_runner._run_one("test", name, parameters, 100)
            for name, parameters in params.items()
        )
        optimized_runner = ExperimentRunner(tiny_events(), config)
        full_runs = []
        original_compactor = ExperimentRunner._compact_checkpoint_run
        def capture_before_compaction(run, path):
            full_runs.append(run)
            return original_compactor(run, path)
        with patch.object(
            optimized_runner, "_build_warmup_artifact",
            wraps=optimized_runner._build_warmup_artifact,
        ) as warmup_fit, patch.object(
            optimized_runner, "_compact_checkpoint_run", side_effect=capture_before_compaction
        ):
            optimized = optimized_runner.run_final_test(
                Path(self._temporary_root.name) / "equivalence-checkpoints"
            )
        warmup_fit.assert_called_once_with(100)
        self.assertEqual(len(optimized), 5)
        for original, reused in zip(reference, full_runs):
            self.assertEqual(original.scaler_digest, reused.scaler_digest)
            self.assertEqual(original.candidate_pool_digest, reused.candidate_pool_digest)
            self.assertEqual(dict(original.metrics), dict(reused.metrics))
            self.assertEqual(original.warmup, reused.warmup)
            self.assertEqual(original.simulation.decisions, reused.simulation.decisions)
            self.assertEqual(
                original.simulation.feedback_scheduled,
                reused.simulation.feedback_scheduled,
            )
            self.assertEqual(
                original.simulation.feedback_applied,
                reused.simulation.feedback_applied,
            )

    def test_final_test_resume_skips_verified_runs_without_duplicates(self):
        params = {
            "random": {"random_seed": 42}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.05, "random_seed": 42},
            "linucb": {"alpha": 2.0},
        }
        config = validation_config(
            mode="final_test", final_parameters=params, candidate_pool_depths=(100,),
            results_directory=str(Path(self._temporary_root.name) / "final-results"),
        )
        checkpoint_directory = Path(self._temporary_root.name) / "resume-final"
        first = ExperimentRunner(tiny_events(), config)
        completed = first.run_final_test(checkpoint_directory)
        self.assertEqual(len(tuple(checkpoint_directory.glob("*.checkpoint.json"))), 5)
        resumed = ExperimentRunner(tiny_events(), config)
        with patch.object(resumed, "_build_warmup_artifact") as rebuild, patch.object(
            resumed, "_run_one"
        ) as rerun:
            restored = resumed.run_final_test(checkpoint_directory)
        rebuild.assert_not_called()
        rerun.assert_not_called()
        self.assertEqual(resumed.resumed_run_count, 5)
        self.assertEqual([run.run_id for run in completed], [run.run_id for run in restored])
        for before, after in zip(completed, restored):
            self.assertEqual(dict(before.metrics), dict(after.metrics))
            self.assertEqual(before.candidate_pool_digest, after.candidate_pool_digest)
            self.assertEqual(before.scaler_digest, after.scaler_digest)
            self.assertEqual(Path(before.details_path).read_bytes(), Path(after.details_path).read_bytes())
        self.assertEqual(len({run.run_id for run in restored}), 5)

    def test_final_test_resumes_after_one_completed_configuration(self):
        params = {
            "random": {"random_seed": 42}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.05, "random_seed": 42},
            "linucb": {"alpha": 2.0},
        }
        config = validation_config(
            mode="final_test", final_parameters=params, candidate_pool_depths=(100,),
            results_directory=str(Path(self._temporary_root.name) / "final-results"),
        )
        checkpoint_directory = Path(self._temporary_root.name) / "interrupted-final"
        interrupted = ExperimentRunner(tiny_events(), config)
        real_run_one = interrupted._run_one
        calls = 0
        def fail_after_first(phase, policy_name, policy_parameters, depth, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption after first checkpoint")
            return real_run_one(phase, policy_name, policy_parameters, depth, **kwargs)
        with patch.object(interrupted, "_run_one", side_effect=fail_after_first):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                interrupted.run_final_test(checkpoint_directory)
        self.assertEqual(len(tuple(checkpoint_directory.glob("*.checkpoint.json"))), 1)

        resumed = ExperimentRunner(tiny_events(), config)
        runs = resumed.run_final_test(checkpoint_directory)
        self.assertEqual(resumed.resumed_run_count, 1)
        self.assertEqual(len(runs), 5)
        self.assertEqual(len({run.run_id for run in runs}), 5)
        self.assertEqual(len(tuple(checkpoint_directory.glob("*.checkpoint.json"))), 5)

        reference = ExperimentRunner(tiny_events(), config)
        reference_runs = reference.run_final_test(
            Path(self._temporary_root.name) / "uninterrupted-final"
        )
        for actual, expected in zip(runs, reference_runs):
            self.assertEqual(actual.run_id, expected.run_id)
            self.assertEqual(actual.metrics, expected.metrics)
            self.assertEqual(actual.candidate_pool_digest, expected.candidate_pool_digest)
            self.assertEqual(actual.scaler_digest, expected.scaler_digest)
            self.assertEqual(
                json.loads(Path(actual.details_path).read_text(encoding="utf-8"))["decisions"],
                json.loads(Path(expected.details_path).read_text(encoding="utf-8"))["decisions"],
            )

    def test_final_test_rejects_corrupt_or_incompatible_checkpoints(self):
        params = {
            "random": {"random_seed": 42}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.05, "random_seed": 42},
            "linucb": {"alpha": 2.0},
        }
        config = validation_config(
            mode="final_test", final_parameters=params, candidate_pool_depths=(100,),
            results_directory=str(Path(self._temporary_root.name) / "final-results"),
        )
        checkpoint_directory = Path(self._temporary_root.name) / "corrupt-final"
        ExperimentRunner(tiny_events(), config).run_final_test(checkpoint_directory)
        marker = next(checkpoint_directory.glob("*.checkpoint.json"))
        detail = checkpoint_directory / (marker.name.removesuffix(".checkpoint.json") + ".jsonl")
        detail.write_text(detail.read_text(encoding="utf-8") + "corruption", encoding="utf-8")
        resumed = ExperimentRunner(tiny_events(), config)
        with self.assertRaisesRegex(CheckpointError, "checksum mismatch"):
            resumed.run_final_test(checkpoint_directory)

    def test_final_test_quarantines_unmarked_jsonl_and_reruns_that_configuration(self):
        params = {
            "random": {"random_seed": 42}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.05, "random_seed": 42},
            "linucb": {"alpha": 2.0},
        }
        config = validation_config(
            mode="final_test", final_parameters=params, candidate_pool_depths=(100,),
            results_directory=str(Path(self._temporary_root.name) / "final-results"),
        )
        checkpoint_directory = Path(self._temporary_root.name) / "orphan-final"
        ExperimentRunner(tiny_events(), config).run_final_test(checkpoint_directory)
        marker = next(checkpoint_directory.glob("*.checkpoint.json"))
        marker.unlink()
        resumed = ExperimentRunner(tiny_events(), config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runs = resumed.run_final_test(checkpoint_directory)
        self.assertEqual(len(runs), 5)
        self.assertEqual(resumed.resumed_run_count, 4)
        self.assertTrue(any("quarantined uncommitted" in str(item.message) for item in caught))
        self.assertEqual(len(tuple((checkpoint_directory / ".incomplete").glob("*.uncommitted"))), 1)
        self.assertEqual(len(tuple(checkpoint_directory.glob("*.checkpoint.json"))), 5)

    def test_final_test_rejects_corrupt_warmup_artifact(self):
        params = {
            "random": {"random_seed": 42}, "popularity": {}, "greedy": {},
            "epsilon_greedy": {"epsilon": 0.05, "random_seed": 42},
            "linucb": {"alpha": 2.0},
        }
        config = validation_config(
            mode="final_test", final_parameters=params, candidate_pool_depths=(100,),
            results_directory=str(Path(self._temporary_root.name) / "final-results"),
        )
        checkpoint_directory = Path(self._temporary_root.name) / "corrupt-warmup"
        ExperimentRunner(tiny_events(), config).run_final_test(checkpoint_directory)
        artifact = checkpoint_directory / "warmup-N100.json"
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["scaler_mean"][0] += 1.0
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        resumed = ExperimentRunner(tiny_events(), config)
        with self.assertRaisesRegex(CheckpointError, "artifact checksum mismatch"):
            resumed.run_final_test(checkpoint_directory)


class FrozenSelectionArtifactTests(unittest.TestCase):
    def test_frozen_final_config_and_manifest_hashes(self):
        root = Path(__file__).resolve().parents[1]
        config = ExperimentConfig.from_json(root / "experiments/configs/final_test.json")
        manifest = json.loads(
            (root / "experiments/configs/final_selection_manifest.json").read_text(encoding="utf-8")
        )
        digest = manifest.pop("manifest_sha256")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), digest)
        self.assertEqual(config.configuration_hash, manifest["final_test_config_hash"])
        self.assertEqual(config.mode, "final_test")
        self.assertEqual(config.candidate_pool_depths, (100,))
        self.assertEqual(config.k_values, (5, 10, 20))
        self.assertEqual(config.final_parameters["linucb"]["alpha"], 2.0)
        self.assertEqual(config.final_parameters["epsilon_greedy"]["epsilon"], 0.05)
        self.assertEqual(manifest["protocol_commit"], "c7167312f55fdacd86e4183bfd80a3d1b02d5b55")
        self.assertEqual(manifest["experiment_commit"], "f67c7d9bdf2dfa24de0fe14221adb41c6f1e33a8")
        self.assertEqual(
            hashlib.sha256((root / "experiments/configs/final_test.json").read_bytes()).hexdigest(),
            manifest["final_test_config_file_sha256"],
        )
        self.assertEqual(_experiment_code_digest(root), manifest["final_test_runner_source_digest"])


class ValidationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary_root = tempfile.TemporaryDirectory()
        cls.config = validation_config(
            candidate_pool_depths=(50, 100, 200),
            results_directory=str(Path(cls._temporary_root.name) / "results"),
        )
        cls.runner = ExperimentRunner(tiny_events(), cls.config)
        cls.checkpoint_directory = Path(cls._temporary_root.name) / "checkpoints"
        cls.runs = cls.runner.run_validation_sweep(cls.checkpoint_directory)

    @classmethod
    def tearDownClass(cls):
        cls._temporary_root.cleanup()

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
    def test_workspace_cache_directory_is_created_under_ignored_temp_root(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            cache_root = _workspace_cache_directory(project_root)
            self.assertEqual(
                cache_root,
                project_root / ".pytest-tmp" / "lastfm-sqlite-cache",
            )
            self.assertTrue(cache_root.is_dir())

    def test_loader_creates_cache_parent_and_removes_sqlite_files_on_close(self):
        content = "u1\t2020-01-01T00:00:00Z\ta1\tArtist\tt1\tTrack\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plays.tsv"
            cache_root = root / ".pytest-tmp" / "lastfm-sqlite-cache"
            source.write_text(content, encoding="utf-8")

            events = load_lastfm_tsv(source, temporary_directory=cache_root)
            database_path = events.database_path
            self.assertTrue(cache_root.is_dir())
            self.assertTrue(database_path.is_file())
            self.assertEqual(len(events), 1)
            self.assertEqual(len(tuple(events)), 1)
            events.close()

            self.assertFalse(database_path.exists())
            self.assertEqual(tuple(cache_root.iterdir()), ())

    def test_loader_cleans_cache_after_parse_failure(self):
        content = "u1\t2020-01-01T00:00:00Z\ta1\tArtist\tt1\tTrack\textra\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "malformed.tsv"
            cache_root = root / ".pytest-tmp" / "lastfm-sqlite-cache"
            source.write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected 6"):
                load_lastfm_tsv(source, temporary_directory=cache_root)

            self.assertTrue(cache_root.is_dir())
            self.assertEqual(tuple(cache_root.iterdir()), ())

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
                fallback_item, fallback_artist = resolve_event_identity_keys(None, "Fallback Artist", None, "Track One")
                self.assertEqual(events[0].artist_key, fallback_artist)
                self.assertEqual(events[0].item_key, fallback_item)
                self.assertEqual(events[1].item_key, resolve_event_identity_keys("a2", "Artist Two", "t2", "Track Two")[0])
            finally:
                events.close()
            self.assertFalse(events.database_path.exists())

    def test_double_quotes_are_literal_in_artist_and_track_names(self):
        content = (
            'u1\t2020-01-02T00:00:00Z\ta1\tArtist "Quoted" Name\tt1\t"’Z\n'
            'u2\t2020-01-01T00:00:00Z\ta2\tNext Artist\tt2\tNext Track\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "plays.tsv"
            source.write_text(content, encoding="utf-8")
            events = load_lastfm_tsv(source, temporary_directory=directory)
            try:
                self.assertEqual(len(events), 2)
                # The second physical row remains its own event; quotes do not
                # put the TSV reader into a multiline CSV quoting state.
                self.assertEqual(events[0].user_id, "u2")
                self.assertEqual(events[1].user_id, "u1")
                expected_item, expected_artist = resolve_event_identity_keys("a1", 'Artist "Quoted" Name', "t1", '"’Z')
                self.assertEqual(events[1].artist_key, expected_artist)
                self.assertEqual(events[1].item_key, expected_item)
            finally:
                events.close()

    def test_empty_ids_remain_empty_inputs_to_identity_fallback(self):
        content = "u1\t2020-01-01T00:00:00Z\t\tArtist Name\t\tTrack Name\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "plays.tsv"
            source.write_text(content, encoding="utf-8")
            events = load_lastfm_tsv(source, temporary_directory=directory)
            try:
                self.assertEqual(len(events), 1)
                expected_item, expected_artist = resolve_event_identity_keys(None, "Artist Name", None, "Track Name")
                self.assertEqual(events[0].artist_key, expected_artist)
                self.assertEqual(events[0].item_key, expected_item)
            finally:
                events.close()

    def test_missing_track_id_fallback_uses_artist_id_without_artist_name(self):
        content = "u1\t2020-01-01T00:00:00Z\ta1\t\t\tTrack Name\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "plays.tsv"
            source.write_text(content, encoding="utf-8")
            events = load_lastfm_tsv(source, temporary_directory=directory)
            try:
                expected_item, expected_artist = resolve_event_identity_keys(
                    "a1", None, None, "Track Name"
                )
                self.assertEqual(events[0].artist_key, expected_artist)
                self.assertEqual(events[0].item_key, expected_item)
            finally:
                events.close()

    def test_rejects_physical_row_with_wrong_tab_delimited_column_count(self):
        content = (
            "u1\t2020-01-01T00:00:00Z\ta1\tArtist\tt1\tTrack\n"
            "u2\t2020-01-02T00:00:00Z\ta2\tArtist\tt2\tTrack\textra\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "plays.tsv"
            source.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"TSV row 2 has 7 columns; expected 6"):
                load_lastfm_tsv(source, temporary_directory=directory)


class ExperimentCliCleanupTests(unittest.TestCase):
    def _run_cli_with_mocks(self, runner, events):
        config = validation_config()
        with (
            patch("sys.argv", ["music-discovery", "experiments/config.json"]),
            patch("src.experiments.cli.ExperimentConfig.from_json", return_value=config),
            patch("src.experiments.cli._workspace_cache_directory"),
            patch("src.experiments.cli.load_lastfm_tsv", return_value=events),
            patch("src.experiments.cli.ExperimentRunner", return_value=runner),
        ):
            from src.experiments.cli import main

            main()

    def test_runner_close_error_still_closes_events_and_propagates_runner_error(self):
        runner = MagicMock()
        runner.run_validation_sweep.return_value = ()
        runner.eligible_users = ()
        runner.write_results.return_value = (Path("runs.csv"), Path("runs.jsonl"))
        runner.close.side_effect = RuntimeError("runner close failed")
        events = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "runner close failed"):
            self._run_cli_with_mocks(runner, events)

        runner.close.assert_called_once_with()
        events.close.assert_called_once_with()

    def test_operation_error_remains_primary_when_runner_close_also_fails(self):
        runner = MagicMock()
        runner.run_validation_sweep.side_effect = ValueError("simulation failed")
        runner.close.side_effect = RuntimeError("runner close failed")
        events = MagicMock()
        events.close.side_effect = OSError("events close failed")

        with self.assertRaisesRegex(ValueError, "simulation failed") as raised:
            self._run_cli_with_mocks(runner, events)

        runner.close.assert_called_once_with()
        events.close.assert_called_once_with()
        cleanup_error = raised.exception.__cause__
        self.assertEqual(len(cleanup_error.errors), 2)
        self.assertRegex(str(cleanup_error.errors[0]), "runner close failed")
        self.assertRegex(str(cleanup_error.errors[1]), "events close failed")


if __name__ == "__main__":
    unittest.main()
