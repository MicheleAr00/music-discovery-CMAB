# Held-out test procedure

The pre-test selection is frozen in `configs/final_selection_manifest.json`.
The one-shot configuration is `configs/final_test.json`; it keeps `N=100`,
`K=(5, 10, 20)`, seed 42, and the official temporal protocol. Do not edit its
parameters after test execution begins.

## Provenance gate

Before starting, confirm that the working tree is clean and that the code
commit and experiment code digest recorded in the manifest match the checkout.
Confirm the source dataset digest is the manifest's Last.fm 1K digest. Keep the
held-out output and checkpoint folders outside Git; the default
`experiments/results/` tree is ignored.

## Start and resume

Run from the project root with the project's Python environment:

```powershell
.\.venv\Scripts\python.exe -u -m src.experiments.cli experiments/configs/final_test.json --checkpoint-directory experiments/results/checkpoints/final-test-044f9269d4e3-5b076e7a1cb3
```

To resume, use the exact same configuration, dataset, code checkout, and
checkpoint directory. A checkpoint is reusable only when its completion
marker, JSONL SHA-256, dataset/configuration/code identity, policy parameters,
eligible cohort, scaler digest, and candidate-pool digest all validate. A
partial JSONL without a completion marker, or a stale temporary file, is moved
to a checksummed `.incomplete/` quarantine and that configuration is rerun. A
corrupt completion marker, a checksum mismatch, or an incompatible identity
causes an explicit error; it is never silently treated as a completed run.

## Execution semantics

The runner fits and records the warm-up scaler once. Each policy then receives
a fresh temporal state and candidate generator restored by chronological event
replay, plus a fresh context builder and policy. Each completed policy is
checkpointed before the next begins. The final CSV and JSONL are atomically
replaced from the five verified checkpoint records. No validation learner
state or feedback is carried into test, and no test result is used to select
parameters.

Expected aggregate outputs are under
`experiments/results/test/<final-config-hash-prefix>-<dataset-hash-prefix>/`.
Do not start a second process against the same checkpoint directory while one
run is active.
