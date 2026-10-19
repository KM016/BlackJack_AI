# Verification

Run the following from the repository root in the documented environment:

```bash
python -B -m research.verification verify
python -B -m research.analysis
python -B -m research.notebook
python -B -m research.notebook --execute
python -B -m pytest tests -q -p no:cacheprovider
git diff --check
git status --short
git diff -- README.md
```

These commands check the completed experiment and recompute analysis from saved
session records. Notebook execution refreshes its derived outputs. No final
blackjack sessions or frozen DQN training runs are required.

## What is checked

- Both retained DQNs and matching settings load with their original checkpoint
  and settings hashes. Extended DQN 3 remains primary; Original DQN 1 remains
  exploratory.
- The original selection record, plans, calibration metadata/report, evaluator
  snapshot, experiment metadata, sessions and summary match their saved hashes.
  All 19 calibration-source hashes retain their historical linkage.
- `results/experiment_metadata.json` retains the exact original final-manifest
  bytes and SHA-256. `research/evidence/asset_mapping.json` records the new paths
  as a derived mapping, without changing historical file identities.
- The compact selection summary, the original candidate order and the unchanged
  tie rule reproduce Extended DQN 3 as the winner. Detailed selection-session
  records remain in the verified archive.
- All 2,100 strategy-sessions are unique, paired across seven policies and cover
  the original 300 seeds per policy and 500-opportunity horizon. Session wealth,
  log growth, betting counts, control outcomes and original summary reconcile.
- The tested analysis reproduces the primary paired mean log-growth difference
  −0.017444627584 and 95% CI [−0.035090337022, −0.001425544573]. Narrow floating
  tolerances cover rounding differences in derived statistics only. Original
  checksums, seed identities, labels and accounting requirements remain strict.
- The notebook parses, has all nine analytical sections, contains five rendered
  Matplotlib figures and no execution errors, and reproduces the primary result.
  An automated test executes it from a clean source copy using retained files.
- Required compact models and results are available without the external raw
  datasets or a local `.venv/`. Caches, development outputs and environments are
  excluded from Git; the test suite also checks maintained imports and paths.

The 1,050,000 original raw opportunities were previously streamed and reconciled
to session accounting, reward sums, drawdown and betting activity. Raw data and
that audit's compact record are in verified external archives; they are not
required by a clean checkout. `asset_mapping.json` checks their recorded
identities, not the bytes of an unavailable archive.

[Reproducibility](reproducibility.md) describes setup, execution and recovery.
The complete lock records the tested Python 3.14.7/macOS arm64 environment,
not a guarantee that identical dependency wheels exist on every platform.
The final task handoff records the actual suite totals and environment checks.

## Owner handoff

`README.md` is reserved for the owner; [README_handoff.md](README_handoff.md)
contains the component explanations, mathematics, source references and concise
results. The notebook is the complete statistical presentation. Keep `docs/`
until README writing is finished.

The only unresolved distribution questions identified here are fixture
redistribution permission and PyDealer/project licence compatibility; see
[attribution.md](attribution.md). Preserve necessary notices in a retained
location before later removing `docs/`. No project licence has been invented.
No commit, push, merge, branch deletion or history rewrite is part of this work.
