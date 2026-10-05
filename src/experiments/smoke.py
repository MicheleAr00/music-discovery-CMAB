"""Small synthetic end-to-end smoke run for all interchangeable policies."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from src.data.state import PlayEvent
from src.experiments.config import ExperimentConfig, TemporalProtocol
from src.experiments.runner import ExperimentRun, ExperimentRunner


def synthetic_smoke_inputs() -> tuple[tuple[PlayEvent, ...], ExperimentConfig]:
    """Return a compact deterministic dataset and a validation-sweep config."""
    warm = datetime(2020, 1, 1, 10)
    validation = datetime(2020, 1, 10, 10)
    test = datetime(2020, 1, 20, 10)
    events = (
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
        candidate_pool_depths=(50,),
        k_values=(5, 10, 20),
        alpha_values=(0.25,),
        epsilon_values=(0.1,),
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
    if len({run.candidate_pool_digest for run in runs}) != 1:
        raise RuntimeError("synthetic smoke policies did not share the same candidate pools")
    if len({run.scaler_digest for run in runs}) != 1:
        raise RuntimeError("synthetic smoke policies did not share the same warm-up scaler")
    if output_directory is not None:
        runner.write_results(runs, output_directory)
    return runs
