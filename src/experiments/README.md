# Experiment Runner

`ExperimentConfig` describes a validation sweep or a manually configured
final-test run. It serializes to JSON, hashes its canonical contents, and
keeps the frozen warm-up/validation/test dates, seven-day reward horizon,
ten-minute session gap, pool depths, K values, and policy sweep ranges in one
place.

```python
from src.experiments import ExperimentConfig, ExperimentRunner
from src.experiments.dataset import load_lastfm_tsv

config = ExperimentConfig.from_json("experiments/configs/validation.json")
events = load_lastfm_tsv(config.dataset_path)
runner = None
try:
    runner = ExperimentRunner(events, config)
    runs = runner.run_validation_sweep()
    runner.write_results(runs)
finally:
    if runner is not None:
        runner.close()
    events.close()
```

Validation sweeps keep only compact run summaries in memory. Full per-decision
records are streamed to temporary files and copied into `runs.jsonl` by
`write_results()`. Keep the runner open until results have been written, then
call `runner.close()` to remove its temporary detail files.

The Last.fm TSV loader makes a disk-backed chronological SQLite cache because
the bundled source file is reverse ordered. It resolves missing track/artist
IDs with the identity key convention already used by state and candidate
generation. The cache is deleted by `close()`.

Eligibility is calculated once and shared by every run. The frozen
`N_train` unit is `events` (warm-up listens) and the config rejects any other
unit. This unit is recorded in every result.
Validation/test usable counts include only session decision points whose
complete reward horizon fits inside that phase and whose candidate pool is
nonempty at every configured pool depth. This common intersection preserves
the eligible cohort across pool-depth comparisons. The configured `N_min`
threshold is then applied to all three counts. Empty-pool records remain in
the simulator audit log but are excluded from metric aggregation.

The primary environment is fixed a priori at `N=100`. Validation may also run
`N=50` and `N=200` for descriptive sensitivity analysis; those results never
select a different primary N. For each policy/parameter combination, the
runner rebuilds an independent simulator and scaler from the same events. It
asserts that the ordered candidate pools and fitted warm-up scaler have
matching SHA-256 digests for all policies at the same pool depth.
Hyperparameter values never affect candidate generation. Results include
flattened `runs.csv` and detailed `runs.jsonl` records under
`experiments/results/<phase>/<config-hash>-<data-hash>/`.

`select_validation_parameters(runs)` selects LinUCB alpha and epsilon-greedy
epsilon from validation rows at `N=100` only, maximizing `NDCG@10`. Exact ties
are resolved by higher Average Reward, then the lower parameter value. The
selector rejects test-phase rows. Pass the result to
`make_final_test_config(selection)` to create a test configuration with `N=100`
for every policy and `K=(5, 10, 20)` preserved as evaluation cutoffs.
`rank_validation_results()` orders rows by the same criterion; request N=50
or N=200 explicitly only to inspect sensitivity results. `run_final_test()`
requires one common `N=100` configuration and can be called only once on a
runner instance. It constructs fresh per-policy simulators, fits only on
warm-up, advances observable state to the test period, and resets LinUCB before
test decisions. No validation learner state is carried into test.

No full-dataset experiment is started by importing these modules or loading a
configuration. The CLI runs only when explicitly invoked with
`python -m src.experiments.cli <config.json>`. Verify the full pipeline without
opening the dataset using `python -m src.experiments.cli --smoke`.
