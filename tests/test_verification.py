"""Check publication relocation and historical provenance without simulations."""

import json
import shutil
from pathlib import Path

import pytest

from research.verification import MODEL_FOLDERS, file_hash, resolve_model_folder, verify_publication


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def publication_copy(tmp_path):
    manifest = json.loads((ROOT / "research/evidence/asset_mapping.json").read_text())
    calibration = json.loads((ROOT / "models/calibration_model.json").read_text())
    relatives = set(manifest["inputs"]) | set(calibration["metadata"]["code_hashes"])
    relatives.add("research/evidence/asset_mapping.json")
    for relative in relatives:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def update_publication(root, edit):
    path = root / "research/evidence/asset_mapping.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def test_publication_is_self_contained():
    result = verify_publication(ROOT)
    assert result["checks"]["frozen_models_verified"] == 2
    assert result["checks"]["calibration_source_files_verified"] == 19
    assert result["checks"]["historical_evaluator_source_verified"] is True
    assert result["checks"]["selection_winner_reproduced"] == "extended_3"
    assert result["checks"]["archived_raw_hash_links_verified"] == 3
    assert result["checks"]["archive_contents_rehashed"] is False
    assert result["selection"]["selected_candidate"] == "extended_3"
    assert result["manifest"]["exploratory_candidate"] == "original_1"
    assert all(path.is_file() for path in result["paths"].values())


def test_renamed_metadata_preserves_original_file_identity():
    from research.verification import EXPERIMENT_METADATA_SHA256, HISTORICAL_PATHS

    assert HISTORICAL_PATHS["outputs/final_evaluation_7/manifest.json"] == "results/experiment_metadata.json"
    assert file_hash(ROOT / "results/experiment_metadata.json") == EXPERIMENT_METADATA_SHA256


def test_verification_uses_requested_checkout(publication_copy):
    assert not (publication_copy / "outputs").exists()
    result = verify_publication(publication_copy)
    assert result["paths"]["models/selection.json"].parent.parent == publication_copy
    for historical, relative in MODEL_FOLDERS.items():
        assert resolve_model_folder(historical, publication_copy) == publication_copy / relative


@pytest.mark.parametrize("historical", [
    "outputs/dqn/train_2", "outputs/dqn_extended/train_1", "models/dqn_primary",
    "../outputs/dqn/train_1", "outputs/dqn/train_1/",
])
def test_resolver_rejects_unknown_identity(historical):
    with pytest.raises(ValueError, match="Unknown frozen model"):
        resolve_model_folder(historical)


@pytest.mark.parametrize("relative", [
    "models/dqn_primary/policy.pt", "models/dqn_exploratory/settings.json",
    "results/sessions.csv", "research/evidence/calibration_metadata.json",
])
def test_corrupted_input_is_rejected(publication_copy, relative):
    path = publication_copy / relative
    path.write_bytes(path.read_bytes() + b"\ncorruption")
    with pytest.raises(ValueError, match="input hash differs"):
        verify_publication(publication_copy)


def test_missing_input_is_rejected(publication_copy):
    (publication_copy / "models/dqn_primary/policy.pt").unlink()
    with pytest.raises(ValueError, match="Missing publication input"):
        verify_publication(publication_copy)


def test_changed_mapping_is_rejected(publication_copy):
    update_publication(publication_copy, lambda m: m["model_folder_mapping"].update({
        "outputs/dqn_extended/train_3": "models/dqn_exploratory",
    }))
    with pytest.raises(ValueError, match="mapping changes a frozen identity"):
        verify_publication(publication_copy)


def test_missing_hash_is_rejected(publication_copy):
    update_publication(publication_copy, lambda m: m["inputs"].pop("models/calibration_model.json"))
    with pytest.raises(ValueError, match="omits a required input"):
        verify_publication(publication_copy)


@pytest.mark.parametrize("relative, message", [
    ("results/sessions.csv", "Historical final-result checksum"),
    ("research/evidence/final_evaluation_source.py.txt", "evaluation_source_sha256"),
    ("models/dqn_primary/policy.pt", "Frozen extended_3 policy.pt provenance"),
])
def test_derived_hash_cannot_replace_historical_provenance(publication_copy, relative, message):
    path = publication_copy / relative
    path.write_bytes(path.read_bytes() + b"\n")
    update_publication(publication_copy, lambda m: m["inputs"].update({relative: file_hash(path)}))
    with pytest.raises(ValueError, match=message):
        verify_publication(publication_copy)


def test_derived_hash_cannot_redefine_original_experiment(publication_copy):
    relative = "results/experiment_metadata.json"
    path = publication_copy / relative
    path.write_bytes(path.read_bytes() + b"\n")
    update_publication(publication_copy, lambda m: m["inputs"].update({relative: file_hash(path)}))
    with pytest.raises(ValueError, match="Original experiment metadata checksum differs"):
        verify_publication(publication_copy)


def test_derived_hash_cannot_redefine_losing_selection_candidate(publication_copy):
    relative = "research/evidence/selection_summary.csv"
    path = publication_copy / relative
    path.write_text(path.read_text().replace("0.017586898492908076", "0.018"))
    update_publication(publication_copy, lambda m: m["inputs"].update({relative: file_hash(path)}))
    with pytest.raises(ValueError, match="Original selection summary checksum differs"):
        verify_publication(publication_copy)


def test_calibration_source_drift_is_rejected(publication_copy):
    source = publication_copy / "blackjack/engine.py"
    source.write_bytes(source.read_bytes() + b"\n# changed source\n")
    with pytest.raises(ValueError, match="Calibration source provenance differs: blackjack/engine.py"):
        verify_publication(publication_copy)


def test_archived_raw_link_is_checked(publication_copy):
    update_publication(publication_copy, lambda m: m["archived_raw_inputs"][
        "outputs/final_evaluation_7/rounds.csv"].update({"sha256": "0" * 64}))
    with pytest.raises(ValueError, match="Archived raw-input provenance differs"):
        verify_publication(publication_copy)


def test_input_path_cannot_traverse_outside_checkout(publication_copy):
    update_publication(publication_copy, lambda m: m["inputs"].update({"../outside.txt": "0" * 64}))
    with pytest.raises(ValueError, match="repository-relative asset path"):
        verify_publication(publication_copy)


def test_asset_symlink_cannot_escape_checkout(publication_copy, tmp_path):
    path = publication_copy / "models/dqn_primary/policy.pt"
    outside = tmp_path.parent / f"{tmp_path.name}-outside-policy.pt"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the checkout"):
        verify_publication(publication_copy)
