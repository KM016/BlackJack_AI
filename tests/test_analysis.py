"""Regression and failure checks for saved-evidence analysis; no simulations."""

import csv
import json
import math
import statistics
from pathlib import Path

import pytest

from research.analysis import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, INITIAL_BANKROLL, SESSION_FIELDS,
    _metric_estimates, _read_csv, analyse, bootstrap_indices, clopper_pearson_interval,
    file_hash, paired_comparison, quantile, strategy_statistics, validate_rounds, validate_sessions,
    validate_summary, wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]

# libm rounding can differ across platforms; keep scientific regressions tight
# without requiring identical last bits from independently computed floats.
FLOAT_TOLERANCE = {"rel": 1e-12, "abs": 1e-15}


@pytest.fixture
def manifest():
    return json.loads((ROOT / "results/experiment_metadata.json").read_text())


@pytest.fixture
def session_rows():
    return _read_csv(ROOT / "results/sessions.csv", SESSION_FIELDS)


def test_saved_sessions_and_summary_reconcile(session_rows, manifest):
    groups = validate_sessions(session_rows, manifest)
    validate_summary(_read_csv(ROOT / "results/summary.csv"), groups)
    assert len(groups) == 7
    assert sum(len(group) for group in groups.values()) == 2100
    assert all(len(group) == 300 for group in groups.values())
    assert sum(row["rounds"] for group in groups.values() for row in group) == 1050000


def test_session_order_does_not_change_pairing(session_rows, manifest):
    forward = validate_sessions(session_rows, manifest)
    backward = validate_sessions(list(reversed(session_rows)), manifest)
    assert forward == backward
    assert [row["seed"] for row in forward["dqn"]] == sorted(manifest["seeds"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_strategy", "wrong_seed"])
def test_bad_session_identities_are_rejected(session_rows, manifest, change):
    if change == "missing":
        session_rows.pop()
    elif change == "duplicate":
        session_rows[-1] = dict(session_rows[0])
    elif change == "wrong_strategy":
        session_rows[0]["strategy"] = "unknown"
    else:
        session_rows[0]["seed"] = "123"
    with pytest.raises(ValueError):
        validate_sessions(session_rows, manifest)


@pytest.mark.parametrize("field,value", [
    ("final_bankroll", "nan"), ("log_growth", "NaN"), ("mean_bet", "inf"),
    ("return_std", "-Infinity"), ("bets", "1.5"), ("rounds", "499"),
    ("net_profit", "700"), ("log_growth", "0.5"), ("max_drawdown", "-0.1"),
    ("mean_bet", "8.1"), ("terminal_reason", "unknown"),
])
def test_invalid_session_measurements_are_rejected(session_rows, manifest, field, value):
    session_rows[0][field] = value
    with pytest.raises(ValueError):
        validate_sessions(session_rows, manifest)


def test_no_bet_baseline_is_checked(session_rows, manifest):
    row = next(row for row in session_rows if row["strategy"] == "never_bet")
    row["max_drawdown"] = "0.01"
    with pytest.raises(ValueError, match="zero-bet session|no-bet baseline"):
        validate_sessions(session_rows, manifest)


def test_session_fields_are_required(session_rows, manifest):
    del session_rows[0]["max_drawdown"]
    with pytest.raises(ValueError, match="columns"):
        validate_sessions(session_rows, manifest)


def test_terminal_loss_requires_at_least_that_drawdown(session_rows, manifest):
    row = next(row for row in session_rows if float(row["final_bankroll"]) < 1000)
    row["max_drawdown"] = "0"
    with pytest.raises(ValueError, match="drawdown is smaller"):
        validate_sessions(session_rows, manifest)


def test_functional_ruin_requires_inability_to_place_minimum_wager(session_rows, manifest):
    session_rows[0]["terminal_reason"] = "no_positive_wager"
    with pytest.raises(ValueError, match="terminal reason"):
        validate_sessions(session_rows, manifest)


def test_summary_must_match_saved_sessions(session_rows, manifest):
    groups = validate_sessions(session_rows, manifest)
    summary = _read_csv(ROOT / "results/summary.csv")
    summary[0]["mean_log_growth"] = "0.5"
    with pytest.raises(ValueError, match="does not reconcile"):
        validate_summary(summary, groups)


def test_summary_nonfinite_and_duplicate_rows_are_rejected(session_rows, manifest):
    groups = validate_sessions(session_rows, manifest)
    summary = _read_csv(ROOT / "results/summary.csv")
    summary[0]["mean_log_growth"] = "NaN"
    with pytest.raises(ValueError):
        validate_summary(summary, groups)
    summary[-1] = dict(summary[0])
    with pytest.raises(ValueError, match="missing or duplicated"):
        validate_summary(summary, groups)


def test_summary_missing_fields_are_rejected(session_rows, manifest):
    groups = validate_sessions(session_rows, manifest)
    summary = _read_csv(ROOT / "results/summary.csv")
    del summary[0]["strategy"]
    with pytest.raises(ValueError, match="columns"):
        validate_summary(summary, groups)


def test_linear_quantile_definition_and_endpoints():
    assert quantile([30, 10, 20, 40], 0) == 10
    assert quantile([30, 10, 20, 40], 1) == 40
    assert quantile([30, 10, 20, 40], 0.25) == 17.5
    assert quantile([2], 0.975) == 2
    with pytest.raises(ValueError):
        quantile([], 0.5)


def test_bootstrap_preserves_pairs_and_is_reproducible():
    indices = bootstrap_indices(3, 200, 123)
    assert indices == bootstrap_indices(3, 200, 123)
    result = paired_comparison([100, -50, 13], [99, -51, 12], indices)
    assert result == {"estimate": 1.0, "ci95": [1.0, 1.0]}
    assert all(len(draw) == 3 for draw in indices)
    assert set(index for draw in indices for index in draw) == {0, 1, 2}
    with pytest.raises(ValueError, match="equal lengths"):
        paired_comparison([1, 2], [1], indices)


def test_conditional_wager_is_weighted_by_bets():
    base = {
        "log_growth": 0.0, "final_bankroll": INITIAL_BANKROLL, "max_drawdown": 0,
        "rounds": 500, "bets": 0, "mean_bet": 0,
    }
    rows = [dict(base, bets=1, mean_bet=8), dict(base, bets=9, mean_bet=2), dict(base)]
    estimates = _metric_estimates(rows)
    assert estimates["mean_wager_conditional_on_bet"] == pytest.approx(2.6)
    assert estimates["participation_rate"] == pytest.approx(10 / 1500)
    assert _metric_estimates([base, base])["mean_wager_conditional_on_bet"] == 0


def test_no_bet_conditional_wager_is_undefined(session_rows, manifest):
    rows = validate_sessions(session_rows, manifest)["never_bet"][:2]
    statistics = strategy_statistics(rows, bootstrap_indices(2, 10))
    assert statistics["mean_wager_conditional_on_bet"] == {"estimate": None, "ci95": None}


def test_lower_tail_statistics_use_independent_sessions():
    rows = [{"log_growth": float(index), "final_bankroll": 1000 + index,
             "max_drawdown": 0.01 * index, "rounds": 500, "bets": 0, "mean_bet": 0}
            for index in range(21)]
    estimates = _metric_estimates(rows)
    # ceil(0.05 * 21) = two sessions, distinct from a quantile-interpolated ES.
    assert estimates["log_growth_lower_tail_mean_05"] == 0.5
    assert estimates["log_growth_q05"] == 1.0
    assert estimates["variance_log_growth"] == pytest.approx(38.5)


def test_zero_ruin_interval_is_nonzero_and_exact():
    lower, upper = clopper_pearson_interval(0, 300)
    assert lower == 0
    assert upper == pytest.approx(0.012220974694293507, **FLOAT_TOLERANCE)
    assert clopper_pearson_interval(300, 300) == pytest.approx([1 - upper, 1])
    assert clopper_pearson_interval(1, 2) == pytest.approx([0.01257911709342506, 0.9874208829065749])
    assert wilson_interval(150, 300)[0] < 0.5 < wilson_interval(150, 300)[1]
    with pytest.raises(ValueError):
        clopper_pearson_interval(301, 300)


def test_primary_regression_matches_independent_saved_data_audit(session_rows, manifest):
    groups = validate_sessions(session_rows, manifest)
    assert BOOTSTRAP_REPLICATES == 2000
    assert BOOTSTRAP_SEED == 25002
    result = paired_comparison(
        [row["log_growth"] for row in groups["dqn"]],
        [row["log_growth"] for row in groups["model_based_log"]], bootstrap_indices(300),
    )
    assert result["estimate"] == pytest.approx(-0.017444627583837562, **FLOAT_TOLERANCE)
    assert result["ci95"] == pytest.approx(
        [-0.035090337022188604, -0.0014255445726873972], **FLOAT_TOLERANCE)
    assert math.expm1(result["estimate"]) == pytest.approx(-0.017293350999905633, **FLOAT_TOLERANCE)
    assert result["ci95"][1] < 0


def test_csv_rejects_duplicate_headers_and_malformed_rows(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x,x\n1,2\n")
    with pytest.raises(ValueError, match="header"):
        _read_csv(path)
    path.write_text("x,y\n1\n")
    with pytest.raises(ValueError, match="malformed"):
        _read_csv(path)


def test_round_audit_rejects_checksum_mismatch(tmp_path, session_rows, manifest):
    path = tmp_path / "rounds.csv"
    path.write_text("altered historical evidence\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_rounds(path, session_rows, manifest)


@pytest.mark.parametrize("field,value", [("bankroll_after", "1009"), ("wager", "8"),
                                          ("log_reward", "NaN"), ("true_count", "inf"),
                                          ("true_count", ""), ("round_status", "absorbed")])
def test_round_audit_checks_accounting_and_finite_values(tmp_path, session_rows, manifest, field, value):
    path = tmp_path / "rounds.csv"
    row = {
        "strategy": "flat", "seed": 270001, "opportunity": 1, "round_status": "complete",
        "true_count": "0", "action_index": 1, "wager": "1", "bankroll_before": "1000",
        "bankroll_after": "1000", "net_profit": "0", "round_return": "0", "log_reward": "0",
    }
    row[field] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    manifest["result_sha256"]["rounds.csv"] = file_hash(path)
    with pytest.raises(ValueError, match="Invalid raw evidence"):
        validate_rounds(path, session_rows, manifest)


def test_full_analysis_schema_and_saved_reproducibility(session_rows, manifest):
    result = analyse(ROOT)
    assert result["primary_comparison"]["estimate"] == pytest.approx(-0.017444627583837562, **FLOAT_TOLERANCE)
    assert result["primary_comparison"]["ci95"] == pytest.approx(
        [-0.035090337022188604, -0.0014255445726873972], **FLOAT_TOLERANCE)
    assert result["primary_comparison"]["zero_in_interval"] is False
    assert result["primary_comparison"]["practical_effect_threshold"] is None
    assert len(result["paired_comparisons"]) == 21
    assert sum(row["role"] == "confirmatory_primary" for row in result["paired_comparisons"]) == 1
    assert result["strategies"]["dqn"]["bets"] == 149888
    assert result["strategies"]["dqn"]["mean_wager_conditional_on_bet"]["estimate"] == pytest.approx(4.758559724594364)
    assert result["strategies"]["never_bet"]["mean_log_growth"] == {"estimate": 0, "ci95": [0, 0]}
    assert result["strategies"]["never_bet"]["functional_ruin"]["ci95"][1] > 0
    assert result["provenance"]["analysis_source_sha256"] == file_hash(ROOT / "research/analysis.py")
    for relative, digest in result["provenance"]["input_sha256"].items():
        assert not Path(relative).is_absolute()
        assert file_hash(ROOT / relative) == digest
    # Reconcile regenerated statistics against historical aggregates and direct
    # session calculations rather than a redundant JSON cache. Discrete
    # identities/counts are exact; only computed floating-point values tolerate
    # negligible libm rounding differences between Python environments.
    groups = validate_sessions(session_rows, manifest)
    assert result["study"]["strategies"] == manifest["strategies"]
    assert result["study"]["session_count"] == manifest["session_count"] == 2100
    assert result["study"]["opportunity_count"] == manifest["opportunity_count"] == 1050000
    assert result["settings"]["bootstrap_replicates"] == BOOTSTRAP_REPLICATES == 2000
    assert result["settings"]["bootstrap_seed"] == BOOTSTRAP_SEED == 25002
    for summary in _read_csv(ROOT / "results/summary.csv"):
        name = summary["strategy"]
        rows = groups[name]
        policy = result["strategies"][name]
        assert policy["sessions"] == int(summary["sessions"]) == 300
        assert policy["bets"] == sum(row["bets"] for row in rows)
        assert policy["sits"] == sum(row["sits"] for row in rows)
        assert policy["opportunities"] == policy["bets"] + policy["sits"] == 150000
        for metric, column in (("mean_log_growth", "mean_log_growth"),
                               ("mean_terminal_bankroll", "mean_final_bankroll"),
                               ("mean_max_drawdown", "mean_max_drawdown"),
                               ("participation_rate", "betting_rate")):
            assert policy[metric]["estimate"] == pytest.approx(float(summary[column]), **FLOAT_TOLERANCE)
        logs = [row["log_growth"] for row in rows]
        assert policy["median_log_growth"]["estimate"] == pytest.approx(
            statistics.median(logs), **FLOAT_TOLERANCE)
        assert policy["std_log_growth"]["estimate"] == pytest.approx(
            statistics.stdev(logs), **FLOAT_TOLERANCE)
        positive = sum(row["net_profit"] > 0 for row in rows)
        assert policy["positive_terminal_return"]["count"] == positive
        assert policy["positive_terminal_return"]["estimate"] == positive / 300
        # Independent algebraic Wilson expression, using the standard normal
        # 97.5th percentile. Its last bits can legitimately vary across libm.
        z, n = 1.959963984540054, 300
        centre = (positive + z * z / 2) / (n + z * z)
        radius = z * math.sqrt(positive * (n - positive) / n + z * z / 4) / (n + z * z)
        assert policy["positive_terminal_return"]["ci95"] == pytest.approx(
            [max(0.0, centre - radius), min(1.0, centre + radius)], **FLOAT_TOLERANCE)
        assert policy["functional_ruin"]["count"] == int(summary["functional_ruin_sessions"]) == 0
        assert policy["functional_ruin"]["ci95"] == pytest.approx(
            [0.0, 1 - 0.025 ** (1 / 300)], **FLOAT_TOLERANCE)
