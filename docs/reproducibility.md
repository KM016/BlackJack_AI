# Reproducibility

The compact repository contains executable research code, two frozen DQNs,
the calibration model, the original selection record and experiment metadata,
and all 2,100 final session records. The
[executed notebook](../notebooks/BlackJackAI_Analysis.ipynb) is the central
presentation of the completed statistical research. It recomputes tables and
five Matplotlib figures from those saved records.

## Install

The resolved environment was tested with **Python 3.14.7 on macOS arm64**.
From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` pins the necessary direct dependencies: PyDealer, PyTorch,
Matplotlib, pytest, nbformat, nbclient and ipykernel. Statistics use Python's
standard library. `requirements-lock.txt` records the complete tested environment,
including transitive versions. Install the lock instead of the direct requirements
to reproduce that platform's resolved versions. Other platforms may need different
wheels or platform dependencies; the macOS/Python lock is not universally portable.

No Jupyter server or nbconvert installation is required for execution. Opening
and browsing the saved notebook on GitHub requires no Python installation.
An interactive notebook interface can be installed separately if desired.
`.venv/` stays local and is excluded from Git. Dependencies are installed from
upstream distributions; their source/binaries are not bundled in this repository.

## Verify and reproduce

```bash
python -B -m research.verification verify
python -B -m research.analysis
python -B -m research.notebook
python -B -m research.notebook --execute
python -B -m pytest tests -q -p no:cacheprovider
```

- `research.verification verify` checks frozen bytes, model identities, seed roles,
  calibration linkage and the original selection decision without simulation.
- `research.analysis` validates the saved data and prints recomputed statistics.
  It is read-only and does not regenerate a statistics cache or standalone reports.
- `research.notebook` validates the saved notebook, expected sections, five
  rendered figures, execution status and primary statistical result.
- `research.notebook --execute` runs the notebook using nbclient and the current
  Python kernel, checks it, then saves refreshed outputs for GitHub viewing.
- The suite covers the engine, policies, mathematical methods, frozen evidence
  and notebook, including execution from a clean source copy without local
  development outputs, archived raw data or `.venv/`.

The notebook executes only analysis of existing sessions. It does not simulate
final seeds, train networks or rewrite historical results. Its figures are
created by notebook cells, with `lightsteelblue` figure backgrounds and
`lemonchiffon` axes backgrounds. There are no duplicate standalone chart exports.
Tiny component tests use temporary fixtures and non-final seeds; they do not
retrain either frozen research checkpoint.

Statistics use 2,000 whole-session paired bootstrap resamples and the disclosed
analysis-only RNG seed 25002. Individual policy intervals are pointwise;
the primary contrast preserves session pairing. Reproduction compares floating
calculations with narrow numerical tolerances where platforms can differ in
rounding. Historical file hashes, seeds, labels and dimensions remain exact.

To load a retained policy without generating simulation outcomes:

```python
from rl.policy import load_policy

primary = load_policy("models/dqn_primary")
exploratory = load_policy("models/dqn_exploratory")
```

## Retained files and historical identities

| Current path | Historical identity or role |
| --- | --- |
| `models/dqn_primary/` | Extended DQN 3, originally `outputs/dqn_extended/train_3/` |
| `models/dqn_exploratory/` | Original DQN 1, originally `outputs/dqn/train_1/` |
| `models/calibration_model.json` | Original fitted calibration model |
| `models/selection.json` | Original formal selection record |
| `results/experiment_metadata.json` | Original `outputs/final_evaluation_7/manifest.json`, unchanged bytes |
| `results/sessions.csv`, `results/summary.csv` | Original final numerical outputs |
| `research/evidence/asset_mapping.json` | Derived mapping of historical identities to retained files |
| `research/evidence/calibration_metadata.json` | Calibration dataset/source provenance |
| `research/evidence/calibration_validation_metadata.json` | Independent validation provenance |
| `research/evidence/calibration_validation_report.json` | Original independent validation report |
| `research/evidence/selection_summary.csv` | Six-candidate summary reproducing the frozen winner |
| `research/evidence/final_evaluation_source.py.txt` | Original evaluator source snapshot |

The original experiment metadata SHA-256 is
`2c003f7fe316029808f6fbc0e966d77e34659e1b0fe8c3891d3474d27bfbd3b5`.
Renaming it did not create a new experiment. Weights/settings, selection,
sessions, summary, plans and original supporting records retain their historical
bytes. Old `outputs/` paths inside historical JSON remain evidence of identity;
they are resolved through the explicit mapping rather than rewritten.

All 19 files recorded in calibration source metadata retain their original
hashes. The current evaluator resolves the retained model locations; its
original snapshot still matches the source hash in the original experiment
metadata. The retained selection summary reproduces the winner using the
original rule; detailed candidate sessions are archived.

## External research archives

The independently located `BlackJackAI-original.tar` preserves the complete
pre-publication tree: raw datasets, all candidate models/settings/logs, original
records, source, diagnostics, Git metadata and environment. Its adjacent
`inventory.json`, `SHA256SUMS`, `verification.json` and `RESTORE.md` record
per-file hashes and recovery instructions. Its archive identity remains in
`research/evidence/asset_mapping.json`.

A separately verified pre-refinement backup, `BlackJackAI-before-notebook.tar`,
preserves the later compact tree,
including reports, figure exports, detailed selection sessions, the raw-round
audit and dependency-licensing inventory removed during the notebook refinement.
These files were archived before removal; the backup SHA-256 is
`2ef7cd989a0fe601e2096b7b78f1e1f025e2aceceea01574fc1835eead9108a9`.
The owner retains both archives
outside the working repository; no public archive download is configured.
They are same-disk local backups, not off-device preservation.

For the original archive, from its directory:

```bash
shasum -a 256 -c SHA256SUMS
mkdir recovery
tar -xf BlackJackAI-original.tar -C recovery
```

Extract into a new empty directory. Do not restore over an active checkout.
Full historical reconstruction requires owner-supplied archives; ordinary
verification, notebook execution and tests do not. The original raw final
`rounds.csv` has 1,050,000 records. It and the calibration CSVs were hashed and
audited before archiving. Compact checks reconcile recorded archived-data
identities; they cannot rehash an archive absent from a clean checkout.

To repeat the raw-round accounting audit after restoring the original archive,
without simulating games:

```bash
python -B -m research.analysis --rounds recovery/BlackJackAI/outputs/final_evaluation_7/rounds.csv
```

Adjust the recovery path to the extraction location. A failed checksum or
accounting check is an error; do not generate replacement outcomes.

## Historical research entry points

Calibration fitting, independent validation, DQN training, candidate selection
and final evaluation source remain readable. Their old generated paths describe
the archived experiment; the public tree intentionally contains only two DQNs
and compact final data. Reconstructing every historical training step needs the
archive as well as the frozen plans.

Do not run `research.final_evaluation --mode final` again, reuse its evaluation
seeds for new simulations, tune models on the final outcomes or reinterpret the
exploratory comparator as the selected primary model.
