import csv
import hashlib
import json
from pathlib import Path
from copy import deepcopy
from decimal import Decimal

import pytest

from research.calibration import collect_session, describe_round, generate_samples
from research.calibration_model import (get_bin, parse_bins, sample_statistics,
    bootstrap_mean_ci, summarise, read_samples, CalibrationModel, fit_model)
from research.study_plan import load_plan, validate_plan, plan_hash
from research.validate_calibration import validate_model


# create small independent sessions for fast integration tests
def make_plan():
    return {
        "version": "blackjack_research_v1", "owner_approved": True,
        "seeds": {"calibration": [711, 712], "calibration_validation": [713, 714],
                  "train_1": [715], "train_2": [716], "train_3": [717],
                  "selection": [718], "evaluation": [719]},
        "calibration": {"rounds_per_session": 10, "bins": ["0"],
                        "min_samples": 2, "min_sessions": 2,
                        "bootstrap_replicates": 100, "bootstrap_seed": 4,
                        "max_ci_width": "20", "min_variance": "0.000000000001"}
    }


def synthetic_rows():
    return [
        {"seed": 711, "status": "complete", "true_count": Decimal("0"), "return": Decimal("-1")},
        {"seed": 711, "status": "complete", "true_count": Decimal("0"), "return": Decimal("1")},
        {"seed": 712, "status": "complete", "true_count": Decimal("0"), "return": Decimal("1")},
        {"seed": 712, "status": "complete", "true_count": Decimal("0"), "return": Decimal("1")},
        {"seed": 711, "status": "void", "true_count": Decimal("0"), "return": None}
    ]


# TEST 1: Check that bin boundaries are assigned correctly
def test_bins():
    boundaries = parse_bins(["-1", "0", "1"])
    assert get_bin("-1.1", boundaries) == 0
    assert get_bin("-1", boundaries) == 1
    assert get_bin("0", boundaries) == 2
    assert get_bin("1", boundaries) == 3
    with pytest.raises(ValueError):
        parse_bins(["1", "0"])


# TEST 2: Check the mean and sample variance
def test_mean_variance():
    mean, variance = sample_statistics([Decimal(-1), Decimal(1), Decimal(1), Decimal(1)])
    assert mean == Decimal("0.5")
    assert variance == Decimal(1)
    assert sample_statistics([]) == (None, None)


# TEST 3: Check that bootstrap resamples entire sessions
def test_bootstrap_sessions():
    cluster = {711: [Decimal(-1), Decimal(-1)], 712: [Decimal(1), Decimal(1)]}
    ci, count = bootstrap_mean_ci(cluster, 100, 4)
    assert count == 100
    assert ci == [Decimal(-1), Decimal(1)]
    assert bootstrap_mean_ci(cluster, 100, 4) == (ci, count)


# TEST 4: Check that summary statistics exclude refunded rounds
def test_summarise():
    summary = summarise(synthetic_rows(), [711, 712], "calibration", make_plan()["calibration"])
    item = summary["bins"][1]
    assert item["count"] == 4
    assert item["mean"] == "0.5"
    assert Decimal(item["variance"]) == Decimal("1")
    assert item["sessions"] == 2
    assert item["supported"] is True
    assert summary["void_rounds"] == 1
    assert sum(x["count"] for x in item["distribution"]) == 4


# TEST 5: Check that unsupported bins do not produce bets based on guesses
def test_unsupported_bin():
    plan = make_plan()
    plan["calibration"]["min_samples"] = 5
    summary = summarise(synthetic_rows(), [711, 712], "calibration", plan["calibration"])
    assert summary["bins"][1]["supported"] is False
    assert "insufficient_samples" in summary["bins"][1]["unsupported_reasons"]


# TEST 6: Check that seed roles cannot overlap
def test_seed_overlap():
    plan = make_plan()
    validate_plan(plan)
    plan["seeds"]["evaluation"] = [711]
    with pytest.raises(ValueError, match="overlap"):
        validate_plan(plan)


# TEST 7: Check that research plans work without owner approval
def test_plan_without_approval(tmp_path):
    plan = make_plan()
    plan["owner_approved"] = False
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert load_plan(path) == plan
    metadata = generate_samples(tmp_path / "data", "calibration", plan)
    assert metadata["attempts"] == 20


# TEST 8: Check that real engine rounds produce valid calibration records
def test_collect_session():
    rows = list(collect_session(711, "calibration", 10))
    assert len(rows) == 10
    assert [r["round"] for r in rows] == list(range(1, 11))
    assert all(r["initial_wager"] == 1 for r in rows)
    assert all(r["status"] in ("complete", "void") for r in rows)
    assert all(r["return"] == r["net_profit"] for r in rows if r["status"] == "complete")
    assert all(not r["reshuffle_pending"] or r["true_count"] == 0 for r in rows)


# TEST 9: Check that the same seed gives identical calibration data
def test_reproducible_samples():
    first = list(collect_session(711, "calibration", 10))
    second = list(collect_session(711, "calibration", 10))
    assert first == second


# TEST 10: Check that calibration files are hashed and cannot be overwritten
def test_data_hash_and_overwrite(tmp_path):
    plan = make_plan()
    folder = tmp_path / "cal"
    metadata = generate_samples(folder, "calibration", plan)
    assert metadata["attempts"] == 20
    assert len(read_samples(folder, "calibration", plan)[1]) == 20
    with pytest.raises(FileExistsError):
        generate_samples(folder, "calibration", plan)
    with (folder / "calibration_samples.csv").open("a") as stream:
        stream.write("corrupted,record\n")
    with pytest.raises(ValueError, match="hash"):
        read_samples(folder, "calibration", plan)


# TEST 11: Check that the dataset rejects missing rounds
def test_missing_samples(tmp_path):
    plan = make_plan()
    folder = tmp_path / "data"
    generate_samples(folder, "calibration", plan)
    path = folder / "calibration_samples.csv"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n")
    metadata_path = folder / "calibration_metadata.json"
    meta = json.loads(metadata_path.read_text())
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta["attempts"] -= 1
    metadata_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Missing"):
        read_samples(folder, "calibration", plan)


# TEST 12: Check that calibration fits without reading validation results
def test_fit_and_predict(tmp_path):
    plan = make_plan()
    folder = tmp_path / "cal"
    generate_samples(folder, "calibration", plan)
    model_path = tmp_path / "model.json"
    model = fit_model(folder, plan, model_path)
    reloaded = CalibrationModel.load(model_path)
    assert model.get_document() == reloaded.get_document()
    assert len(model.get_document()["summary"]["bins"]) == 2
    prediction = model.predict("0")
    assert prediction == reloaded.predict("0")
    if prediction is not None:
        assert prediction["variance"] > 0
    with pytest.raises(FileExistsError):
        model.save(model_path)


# TEST 13: Check that independent validation never changes the fitted model
def test_independent_validation(tmp_path):
    plan = make_plan()
    calibration = tmp_path / "cal"
    validation = tmp_path / "val"
    generate_samples(calibration, "calibration", plan)
    generate_samples(validation, "calibration_validation", plan)
    model_path = tmp_path / "model.json"
    fit_model(calibration, plan, model_path)
    before = model_path.read_bytes()
    report = validate_model(model_path, validation, plan, tmp_path / "report.json")
    assert model_path.read_bytes() == before
    assert report["validation_seed_count"] == 2
    assert len(report["comparisons"]) == 2
    with pytest.raises(FileExistsError):
        validate_model(model_path, validation, plan, tmp_path / "report.json")


# TEST 14: Check that mismatched plans and data populations are rejected
def test_wrong_plan(tmp_path):
    plan = make_plan()
    folder = tmp_path / "cal"
    generate_samples(folder, "calibration", plan)
    other = deepcopy(plan)
    other["calibration"]["bootstrap_seed"] = 5
    with pytest.raises(ValueError, match="plan"):
        read_samples(folder, "calibration", other)


# TEST 15: Check that simulated code versions are recorded with the data
def test_code_provenance(tmp_path):
    from research.study_plan import source_hashes
    plan = make_plan()
    folder = tmp_path / "data"
    generate_samples(folder, "calibration", plan)
    metadata = json.loads((folder / "calibration_metadata.json").read_text())
    assert metadata["code_hashes"] == source_hashes()

    metadata["code_hashes"]["blackjack/engine.py"] = "tampered"
    (folder / "calibration_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="code has changed"):
        read_samples(folder, "calibration", plan)


# TEST 16: Check that the estimated variance has its own bootstrap interval
def test_variance_uncertainty():
    summary = summarise(synthetic_rows(), [711, 712], "calibration", make_plan()["calibration"])
    item = summary["bins"][1]
    assert item["variance_ci"] is not None
    assert item["variance_bootstrap_coverage"] == 100


# TEST 17: Check that an unsupported true-count bin returns no prediction
def test_model_unsupported_bin(tmp_path):
    plan = make_plan()
    plan["calibration"]["min_samples"] = 100
    folder = tmp_path / "calibration"
    generate_samples(folder, "calibration", plan)
    model = fit_model(folder, plan, tmp_path / "model.json")
    assert model.predict("-1") is None


# TEST 18: Check the approved plan uses the original session length and methodology
def test_original_protocol_with_more_sessions():
    path = Path(__file__).resolve().parents[1] / "research" / "study_plan.json"
    plan = load_plan(path)
    assert len(plan["seeds"]["calibration"]) == 120
    assert len(plan["seeds"]["calibration_validation"]) == 40
    assert plan["calibration"]["rounds_per_session"] == 5000
    assert plan["calibration"]["min_variance"] == "0.000000000001"
    assert plan["calibration"]["bootstrap_seed"] == 25001
    assert plan["calibration"]["bootstrap_replicates"] == 2000


# TEST 19: Check near-zero variance cannot be used as evidence of an edge
def test_original_variance_threshold():
    plan = make_plan()
    rows = [{"seed": seed, "status": "complete", "true_count": Decimal(0),
             "return": value} for seed, value in
            ((711, Decimal("1")), (711, Decimal("1.0000001")),
             (712, Decimal("1")), (712, Decimal("1.0000001")))]
    summary = summarise(rows, [711, 712], "calibration", plan["calibration"])
    assert summary["bins"][1]["supported"] is False
    assert "invalid_variance" in summary["bins"][1]["unsupported_reasons"]
