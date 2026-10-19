"""Resolve publication assets and verify preserved evidence without simulation.

Historical JSON files retain the paths written when the experiments ran. Only
the two explicitly identified frozen checkpoints are mapped to published paths.
The derived asset mapping records relocation; it is not an experimental record
and does not replace any original checksum.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

from research.calibration_model import CalibrationModel
from research.study_plan import load_plan, plan_hash


ROOT = Path(__file__).resolve().parents[1]
# Original outputs/final_evaluation_7/manifest.json, before either relocation.
EXPERIMENT_METADATA_SHA256 = "2c003f7fe316029808f6fbc0e966d77e34659e1b0fe8c3891d3474d27bfbd3b5"
# Original outputs/dqn_selection/summary.csv; all six candidate summaries remain
# historical evidence even where a changed losing score would leave the winner.
SELECTION_SUMMARY_SHA256 = "3d8c06a6587c4024b4f12b55e6d4da977034df2e6a1727c2337e9f7156de7c60"
MODEL_FOLDERS = {
    "outputs/dqn_extended/train_3": "models/dqn_primary",
    "outputs/dqn/train_1": "models/dqn_exploratory",
}
HISTORICAL_PATHS = {
    **{f"{old}/{name}": f"{new}/{name}"
       for old, new in MODEL_FOLDERS.items()
       for name in ("policy.pt", "settings.json")},
    "outputs/calibration_model.json": "models/calibration_model.json",
    "outputs/dqn_selection/selected.json": "models/selection.json",
    "outputs/final_evaluation_7/manifest.json": "results/experiment_metadata.json",
    **{f"outputs/final_evaluation_7/{name}": f"results/{name}"
       for name in ("sessions.csv", "summary.csv")},
    "outputs/calibration/calibration_metadata.json": "research/evidence/calibration_metadata.json",
    "outputs/calibration_validation/calibration_validation_metadata.json": "research/evidence/calibration_validation_metadata.json",
    "outputs/calibration_validation_report.json": "research/evidence/calibration_validation_report.json",
    "outputs/dqn_selection/summary.csv": "research/evidence/selection_summary.csv",
}
STRATEGIES = (
    "flat", "hi_lo", "calibrated_kelly", "model_based_log", "dqn",
    "dqn_runner_up", "never_bet",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(root, relative):
    """Reject absolute/traversing paths and symlinks escaping this checkout."""
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Asset paths must be repository-relative POSIX paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative or not path.parts:
        raise ValueError(f"Invalid repository-relative asset path: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Asset path escapes the checkout: {relative}")
    return resolved


def resolve_model_folder(historical, root=None):
    """Map a known historical identity; never accept an arbitrary replacement."""
    historical = str(historical)
    if historical not in MODEL_FOLDERS:
        raise ValueError(f"Unknown frozen model identity: {historical}")
    root = ROOT if root is None else Path(root).resolve()
    return _relative_path(root, MODEL_FOLDERS[historical])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify_publication(root=Path(".")):
    """Verify shipped bytes and their links to historical records, read-only.

    Returns parsed publication/historical manifests, selection, absolute input
    paths and a checks summary. Raw archive contents are not available in a clean
    checkout: their recorded hashes are reconciled here, not rehashed here.
    """
    root = Path(root).resolve()
    publication = json.loads((root / "research/evidence/asset_mapping.json").read_text())
    _require(publication.get("schema_version") == "blackjack_publication_v1",
             "Unknown publication manifest format")
    _require(publication.get("model_folder_mapping") == MODEL_FOLDERS,
             "Publication model mapping changes a frozen identity")
    _require(publication.get("historical_paths") == HISTORICAL_PATHS,
             "Publication historical path mapping has changed")
    inputs = publication.get("inputs")
    _require(isinstance(inputs, dict), "Publication input hashes are missing")
    required = set(HISTORICAL_PATHS.values()) | {
        "research/study_plan.json", "research/study_plan_dqn_extended.json",
        "research/evidence/final_evaluation_source.py.txt",
    }
    _require(required <= inputs.keys(), "Publication manifest omits a required input")
    paths = {}
    for relative, expected in inputs.items():
        _require(_sha(expected), f"Invalid input SHA-256: {relative}")
        path = _relative_path(root, relative)
        _require(path.is_file(), f"Missing publication input: {relative}")
        _require(file_hash(path) == expected, f"Publication input hash differs: {relative}")
        paths[relative] = path
    _require(inputs["results/experiment_metadata.json"] == EXPERIMENT_METADATA_SHA256,
             "Original experiment metadata checksum differs")
    _require(inputs["research/evidence/selection_summary.csv"] == SELECTION_SUMMARY_SHA256,
             "Original selection summary checksum differs")

    def read(relative):
        return json.loads(paths[relative].read_text())

    final = read("results/experiment_metadata.json")
    selected = read("models/selection.json")
    original_plan = load_plan(paths["research/study_plan.json"])
    plan = load_plan(paths["research/study_plan_dqn_extended.json"])
    _require(final.get("status") == "complete" and final.get("mode") == "final",
             "Saved evaluation is not a completed final experiment")
    _require(final.get("strategies") == list(STRATEGIES), "Final strategy labels changed")
    _require(final.get("seeds") == plan["seeds"]["evaluation"] and
             len(final["seeds"]) == len(set(final["seeds"])) == 300,
             "Final seed allocation differs from the frozen plan")
    _require(final.get("rounds_per_session") == 500 and
             final.get("session_count") == 2100 and
             final.get("opportunity_count") == 1050000,
             "Final experiment dimensions differ from the frozen design")
    _require(selected.get("selected_candidate") == final.get("selected_candidate") == "extended_3"
             and final.get("exploratory_candidate") == "original_1",
             "Frozen primary or exploratory model identity changed")
    _require(selected.get("candidates") == [
        "original_1", "original_2", "original_3", "extended_1", "extended_2", "extended_3"
    ], "Historical selection candidate set changed")
    _require(selected.get("final_evaluation_seeds_used") is False and
             selected.get("selection_seeds") == plan["seeds"]["selection"] and
             len(plan["seeds"]["selection"]) == 30,
             "Historical selection seed isolation differs from the plan")
    _require(selected.get("research_plan_sha256") == plan_hash(plan) and
             selected.get("research_plan_file_sha256") == inputs["research/study_plan_dqn_extended.json"],
             "Historical selection research-plan provenance differs")
    # The compact historical summary is sufficient to reproduce the original
    # winner decision. Detailed candidate sessions remain in the research archive.
    from rl.select import choose_winner

    with paths["research/evidence/selection_summary.csv"].open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {"candidate", "sessions", "mean_log_growth", "mean_final_bankroll",
                  "mean_net_profit", "mean_max_drawdown", "betting_rate"}
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(fields)
                 and set(reader.fieldnames) == fields, "Historical selection summary columns differ")
        summaries = list(reader)
    _require([row.get("candidate") for row in summaries] == selected["candidates"],
             "Historical selection summary candidate order differs")
    try:
        summaries = [{"candidate": row["candidate"], "sessions": int(row["sessions"]),
                      **{key: float(row[key]) for key in fields - {"candidate", "sessions"}}}
                     for row in summaries]
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid historical selection summary values") from error
    _require(all(row["sessions"] == 30 and all(math.isfinite(row[key]) for key in fields
                 - {"candidate", "sessions"}) for row in summaries),
             "Historical selection summary dimensions or values differ")
    winner = choose_winner(summaries)
    _require(winner["candidate"] == selected["selected_candidate"] and
             winner == selected.get("winner_summary"),
             "Historical selection summary does not reproduce the frozen winner")

    links = {
        "selection_sha256": "models/selection.json",
        "plan_sha256": "research/study_plan_dqn_extended.json",
        "calibration_sha256": "models/calibration_model.json",
        "evaluation_source_sha256": "research/evidence/final_evaluation_source.py.txt",
    }
    for key, relative in links.items():
        _require(final.get(key) == inputs[relative], f"Historical final provenance differs: {key}")

    for candidate, old, prefix, run, episodes in (
        ("extended_3", "outputs/dqn_extended/train_3", "selected", 3, 500),
        ("original_1", "outputs/dqn/train_1", "exploratory", 1, 100),
    ):
        record = selected.get("model_files", {}).get(candidate, {})
        _require(record.get("folder") == old and final.get(f"{prefix}_policy_folder") == old,
                 f"Historical folder identity differs: {candidate}")
        if candidate == "extended_3":
            _require(selected.get("selected_policy_folder") == old,
                     "Historical selected policy folder differs")
        new = MODEL_FOLDERS[old]
        for filename, key in (("policy.pt", "policy_sha256"), ("settings.json", "settings_sha256")):
            _require(record.get(key) == final.get(f"{prefix}_{key}") == inputs[f"{new}/{filename}"],
                     f"Frozen {candidate} {filename} provenance differs")
        settings = read(f"{new}/settings.json")
        _require(settings.get("seeds") == plan["seeds"][f"train_{run}"][:episodes] and
                 settings.get("network_seed") == 101 * run and
                 settings.get("environment_steps") == episodes * 500 and
                 settings.get("rounds") == 500 and settings.get("initial_bankroll") == "1000" and
                 settings.get("unit") == "1" and settings.get("epsilon_decay_steps") == 10000 and
                 float(settings.get("reward_scale", 1.0)) == 1.0,
                 f"Frozen {candidate} training settings differ")

    calibration = CalibrationModel.load(paths["models/calibration_model.json"]).get_document()
    metadata = read("research/evidence/calibration_metadata.json")
    validation = read("research/evidence/calibration_validation_metadata.json")
    report = read("research/evidence/calibration_validation_report.json")
    _require(calibration.get("metadata") == metadata and
             calibration.get("role") == "calibration" and
             calibration.get("population") == "two_seats_fixed_one_unit_s17_das" and
             calibration.get("plan_hash") == plan_hash(original_plan) and
             calibration.get("settings") == original_plan["calibration"],
             "Frozen calibration model linkage differs from the original plan")
    for role, record in (("calibration", metadata), ("calibration_validation", validation)):
        _require(record.get("status") == "complete" and record.get("role") == role and
                 record.get("population") == calibration["population"] and
                 record.get("plan_hash") == plan_hash(original_plan) and
                 record.get("seeds") == original_plan["seeds"][role] and
                 record.get("rules", {}).get("decks") == 6 and
                 record.get("code_hashes") == metadata.get("code_hashes"),
                 f"Historical {role} metadata provenance differs")
    code_hashes = metadata.get("code_hashes")
    _require(isinstance(code_hashes, dict) and bool(code_hashes), "Calibration source hashes missing")
    for relative, expected in code_hashes.items():
        _require(_sha(expected) and file_hash(_relative_path(root, relative)) == expected,
                 f"Calibration source provenance differs: {relative}")
    _require(report.get("plan_hash") == plan_hash(original_plan) and
             report.get("calibration_seed_count") == len(metadata["seeds"]) and
             report.get("validation_seed_count") == len(validation["seeds"]),
             "Independent calibration validation report linkage differs")

    for filename in ("sessions.csv", "summary.csv"):
        _require(final.get("result_sha256", {}).get(filename) == inputs[f"results/{filename}"],
                 f"Historical final-result checksum differs: {filename}")
    archived = publication.get("archived_raw_inputs", {})
    expected_raw = {
        "outputs/calibration/calibration_samples.csv": metadata.get("sha256"),
        "outputs/calibration_validation/calibration_validation_samples.csv": validation.get("sha256"),
        "outputs/final_evaluation_7/rounds.csv": final.get("result_sha256", {}).get("rounds.csv"),
    }
    _require(set(archived) == set(expected_raw), "Archived raw-input inventory changed")
    for relative, expected in expected_raw.items():
        record = archived[relative]
        _require(_sha(expected) and record.get("sha256") == expected and
                 type(record.get("bytes")) is int and record["bytes"] > 0 and
                 record.get("archive_member") == f"BlackJackAI/{relative}",
                 f"Archived raw-input provenance differs: {relative}")
    archive = publication.get("archive", {})
    _require(_sha(archive.get("sha256")) and _sha(archive.get("inventory_sha256")),
             "Research archive identification is missing")
    _require(publication.get("study") == {
        "confirmatory_policies": ["flat", "hi_lo", "calibrated_kelly", "model_based_log", "dqn", "never_bet"],
        "exploratory_policy": "dqn_runner_up", "primary_model": "extended_3",
        "exploratory_model": "original_1", "primary_comparison": ["dqn", "model_based_log"],
    }, "Publication study labels differ from historical design")
    return {
        "publication_manifest": publication, "manifest": final, "selection": selected,
        "paths": paths,
        "checks": {
            "shipped_inputs_verified": len(paths), "calibration_source_files_verified": len(code_hashes),
            "frozen_models_verified": 2, "historical_evaluator_source_verified": True,
            "selection_winner_reproduced": winner["candidate"],
            "archived_raw_hash_links_verified": len(expected_raw),
            "archive_contents_rehashed": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(verify_publication(args.root)["checks"], indent=2))


if __name__ == "__main__":
    main()
