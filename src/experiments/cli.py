"""Command-line entry point for configured offline experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.config import ExperimentConfig
from src.experiments.dataset import load_lastfm_tsv
from src.experiments.runner import ExperimentRunner
from src.experiments.smoke import run_synthetic_smoke


class _MultipleCleanupErrors(RuntimeError):
    """Keep every failure when more than one CLI resource fails to close."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = tuple(errors)
        super().__init__(f"multiple errors while closing CLI resources ({len(errors)})")


def _workspace_cache_directory(project_root: Path) -> Path:
    """Create the ignored, project-local parent for temporary SQLite caches."""
    directory = project_root / ".pytest-tmp" / "lastfm-sqlite-cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _close_resources(runner: ExperimentRunner | None, events) -> None:
    """Attempt every close and report failures without dropping any of them."""
    errors: list[BaseException] = []
    if runner is not None:
        try:
            runner.close()
        except BaseException as error:
            errors.append(error)
    try:
        events.close()
    except BaseException as error:
        errors.append(error)

    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise _MultipleCleanupErrors(errors) from errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", help="JSON experiment configuration")
    parser.add_argument("--output-directory", type=Path, help="override configured results directory")
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        help="persistent per-configuration checkpoint directory (defaults under results_directory)",
    )
    parser.add_argument("--smoke", action="store_true", help="run the small synthetic five-policy pipeline")
    args = parser.parse_args()

    if args.smoke:
        runs = run_synthetic_smoke(args.output_directory)
        print(f"Synthetic smoke passed: {len(runs)} policy runs")
        if args.output_directory:
            print(f"Results written under: {args.output_directory}")
        return
    if args.config is None:
        parser.error("provide a configuration file or use --smoke")

    config = ExperimentConfig.from_json(args.config)
    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = args.config.resolve().parent.parent.parent / dataset_path
    project_root = Path(__file__).resolve().parents[2]
    cache_directory = _workspace_cache_directory(project_root)
    events = load_lastfm_tsv(dataset_path, temporary_directory=cache_directory)
    runner = None
    try:
        runner = ExperimentRunner(events, config)
        runs = (
            runner.run_validation_sweep(args.checkpoint_directory)
            if config.mode == "validation_sweep"
            else runner.run_final_test(args.checkpoint_directory)
        )
        csv_path, jsonl_path = runner.write_results(runs, args.output_directory)
        print(f"Eligible users: {len(runner.eligible_users)}")
        print(f"Completed runs: {len(runs)}")
        if config.mode == "validation_sweep":
            print(f"Resumed runs: {runner.resumed_run_count}")
        print(f"Checkpoints: {runner.checkpoint_directory}")
        if config.mode == "final_test":
            print(f"Warm-up fit seconds: {runner.final_test_warmup_fit_seconds}")
        print(f"Results: {csv_path}")
        print(f"Detailed records: {jsonl_path}")
    except BaseException as primary_error:
        try:
            _close_resources(runner, events)
        except BaseException as cleanup_error:
            # Preserve the operation failure as the propagated exception;
            # retain cleanup failure as its explicit cause.
            raise primary_error from cleanup_error
        raise
    else:
        _close_resources(runner, events)


if __name__ == "__main__":
    main()
