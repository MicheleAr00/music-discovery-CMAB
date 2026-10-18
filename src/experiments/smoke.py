"""Small synthetic end-to-end smoke run for all interchangeable policies."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from src.data.state import PlayEvent
from src.experiments.config import (
    DEFAULT_ALPHAS,
    DEFAULT_EPSILONS,
    ExperimentConfig,
    TemporalProtocol,
)
from src.experiments.runner import ExperimentRun, ExperimentRunner


def synthetic_smoke_inputs() -> tuple[tuple[PlayEvent, ...], ExperimentConfig]:
    """Return a compact deterministic dataset and a validation-sweep config."""
    warm = datetime(2020, 1, 1, 10)
    validation = datetime(2020, 1, 10, 10)
    test = datetime(2020, 1, 20, 10)
    events = (
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
    )
    protocol = TemporalProtocol(
        warmup_start=date(2020, 1, 1),
        warmup_end=date(2020, 1, 2),
        validation_start=date(2020, 1, 10),
        validation_end=date(2020, 1, 13),
        test_start=date(2020, 1, 20),
        test_end=date(2020, 1, 23),
        reward_horizon_days=1,
        session_gap_minutes=10,
    )
    config = ExperimentConfig(
        dataset_path="synthetic://experiment-smoke",
        dataset_id="synthetic-smoke-v1",
        mode="validation_sweep",
        temporal=protocol,
        n_min=1,
        n_train_unit="events",
        candidate_pool_depths=(50, 100, 200),
        k_values=(5, 10, 20),
        alpha_values=DEFAULT_ALPHAS,
        epsilon_values=DEFAULT_EPSILONS,
    )
    return tuple(sorted(events, key=lambda event: event.timestamp)), config


def run_synthetic_smoke(
    output_directory: Optional[str | Path] = None,
) -> tuple[ExperimentRun, ...]:
    """Exercise config → policies → simulator → metrics → optional result files."""
    events, config = synthetic_smoke_inputs()
    runner = ExperimentRunner(events, config)
    runs = runner.run_validation_sweep()
    expected_policies = set(config.policies)
    if {run.policy for run in runs} != expected_policies:
        raise RuntimeError("synthetic smoke did not execute every configured policy")
    schemas = {tuple(sorted(run.summary_row(config))) for run in runs}
    if len(schemas) != 1:
        raise RuntimeError("synthetic smoke policies produced different result schemas")
    for depth in config.candidate_pool_depths:
        depth_runs = [run for run in runs if run.candidate_pool_depth == depth]
        if len({run.candidate_pool_digest for run in depth_runs}) != 1:
            raise RuntimeError(f"synthetic smoke policies did not share candidate pools at N={depth}")
    if len({run.scaler_digest for run in runs}) != 1:
        raise RuntimeError("synthetic smoke policies did not share the same warm-up scaler")
    selection = runner.select_validation_parameters(runs)
    final_config = runner.make_final_test_config(selection)
    if final_config.candidate_pool_depths != (100,):
        raise RuntimeError("synthetic smoke did not preserve the primary N=100")
    if set(final_config.final_parameters) != expected_policies:
        raise RuntimeError("synthetic smoke selection omitted a policy")
    if output_directory is not None:
        runner.write_results(runs, output_directory)
    return runs
