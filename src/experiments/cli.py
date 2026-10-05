"""Command-line entry point for configured offline experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.config import ExperimentConfig
from src.experiments.dataset import load_lastfm_tsv
from src.experiments.runner import ExperimentRunner
from src.experiments.smoke import run_synthetic_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", help="JSON experiment configuration")
    parser.add_argument("--output-directory", type=Path, help="override configured results directory")
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
    events = load_lastfm_tsv(dataset_path)
    try:
        runner = ExperimentRunner(events, config)
        runs = (
            runner.run_validation_sweep()
            if config.mode == "validation_sweep"
            else runner.run_final_test()
        )
        csv_path, jsonl_path = runner.write_results(runs, args.output_directory)
        print(f"Eligible users: {len(runner.eligible_users)}")
        print(f"Completed runs: {len(runs)}")
        print(f"Results: {csv_path}")
        print(f"Detailed records: {jsonl_path}")
    finally:
        events.close()


if __name__ == "__main__":
    main()
