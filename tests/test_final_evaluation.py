"""Guard the frozen model and test the final evaluator without opening final seeds."""

import csv
import json
import math
from decimal import Decimal

import pytest

from research.final_evaluation import (
    FINAL_CONFIRMATION, INITIAL_BANKROLL, ROUNDS, SMOKE_SEEDS,
    STRATEGIES, check_frozen_inputs, evaluate,
    make_classic_policy, play_classic, play_dqn, record_round, summarize_session,
)
from rl.policy import load_policy
from rl.select import evaluate_session
from test_model_based_log import make_model


# preserve the original primary comparison with one exploratory runner-up
def test_strategy_list_is_frozen():
    assert STRATEGIES == (
        "flat", "hi_lo", "calibrated_kelly", "model_based_log", "dqn", "dqn_runner_up", "never_bet"
    )
    assert len(SMOKE_SEEDS) == len(set(SMOKE_SEEDS)) == 3


# all no-bet opportunities must contribute zero return and constant bankroll
def test_sitout_and_invalid_profit():
    result = record_round("never_bet", 29, 1, "complete", 0, 0, None, Decimal(1000), Decimal(1000))
    assert result["log_reward"] == 0
    assert result["wager"] == ""
    with pytest.raises(ArithmeticError, match="Sitting out"):
        record_round("never_bet", 29, 1, "complete", 0, 0, None, Decimal(1000), Decimal(1001))


# actual profit is used in log rewards, with no training reward multiplier
def test_reward_and_session_reconciliation():
    rows = [record_round("flat", 29, 1, "complete", 0, 1, Decimal(1),
                         Decimal(1000), Decimal(1001))]
    for number in range(2, 501):
        rows.append(record_round("flat", 29, number, "complete", 0, 0, None,
                                 Decimal(1001), Decimal(1001)))
    s = summarize_session(rows, "flat", 29, "horizon")
    assert s["final_bankroll"] == 1001
    assert s["log_growth"] == pytest.approx(math.log(1.001), abs=1e-14)
    assert s["bets"] == 1
    assert s["sits"] == 499
    assert s["max_drawdown"] == 0


# no code path may open the reserved final seeds without explicit approval
def test_final_requires_consent_before_loading_any_file(tmp_path):
    with pytest.raises(ValueError, match="explicit confirmation"):
        evaluate("not-a-plan", "not-a-selection", "not-a-model", tmp_path / "final", mode="final")
    assert not (tmp_path / "final").exists()
    assert FINAL_CONFIRMATION == "I_UNDERSTAND_FINAL_HOLDOUT"


# each classical strategy is wrapped in the identical legal wager grid
def test_classical_policies_use_same_grid():
    for name in STRATEGIES:
        if name in ("dqn", "dqn_runner_up"):
            continue
        policy = make_classic_policy(name, make_model())
        assert hasattr(policy, "choose_wager")
        assert hasattr(policy, "last_action")


# the selected DQN must reproduce the existing selection evaluator exactly
def test_dqn_evaluation_matches_selection_evaluator():
    network = load_policy("models/dqn_primary").network
    summary, rows = play_dqn(290001, network)
    reference = evaluate_session(network, 290001, "extended_3")
    assert len(rows) == ROUNDS
    assert summary["final_bankroll"] == reference["final_bankroll"]
    assert summary["log_growth"] == reference["log_growth"]
    assert summary["max_drawdown"] == reference["max_drawdown"]
    assert summary["bets"] == reference["bets"]


# the exploratory model must use the same greedy evaluation and row calculations
# its results stay labelled separately from the primary DQN

def test_runner_up_uses_same_evaluator():
    network = load_policy("models/dqn_exploratory").network
    summary, rows = play_dqn(290001, network, name="dqn_runner_up")
    reference = evaluate_session(network, 290001, "original_1")
    assert len(rows) == ROUNDS
    assert all(row["strategy"] == "dqn_runner_up" for row in rows)
    assert summary["strategy"] == "dqn_runner_up"
    assert summary["final_bankroll"] == reference["final_bankroll"]
    assert summary["log_growth"] == reference["log_growth"]
    assert summary["bets"] == reference["bets"]


# the engine, public card counter, calibrated strategies and reference seat work together
def test_classic_short_smoke():
    model = make_model()
    for name in STRATEGIES:
        if name in ("dqn", "dqn_runner_up"):
            continue
        summary, rows = play_classic(name, 290001, model)
        assert len(rows) == ROUNDS
        assert summary["rounds"] == ROUNDS
        assert summary["bets"] + summary["sits"] == ROUNDS
        assert summary["void_rounds"] == 0
        if name == "never_bet":
            assert summary["final_bankroll"] == float(INITIAL_BANKROLL)
            assert summary["log_growth"] == 0


# smoke mode produces the correct output structure without running reserved seeds
# the real full seven-strategy smoke is run separately before final evaluation
def test_smoke_end_to_end_and_refuses_overwrite(tmp_path, monkeypatch):
    import research.final_evaluation as final_eval
    calibration_path = tmp_path / "calibration.json"
    make_model().save(calibration_path)
    output = tmp_path / "smoke"
    calls = []

    def fake_run(name, seed, model):
        calls.append((name, seed))
        # simulation behaviour is tested in the separate real-engine tests above
        rows = [record_round(name, seed, index, "complete", 0, 0, None,
                             INITIAL_BANKROLL, INITIAL_BANKROLL)
                for index in range(1, ROUNDS + 1)]
        return summarize_session(rows, name, seed, "horizon"), rows

    monkeypatch.setattr(final_eval, "SMOKE_SEEDS", (290001,))
    monkeypatch.setattr(final_eval, "play_dqn", lambda seed, network, name="dqn": fake_run(name, seed, network))
    monkeypatch.setattr(final_eval, "play_classic", fake_run)
    result = evaluate(
        "research/study_plan_dqn_extended.json",
        "models/selection.json",
        calibration_path, output, mode="smoke",
    )
    assert result["status"] == "complete"
    assert result["session_count"] == 7
    assert result["opportunity_count"] == 3500
    assert result["exploratory_candidate"] == "original_1"
    assert "smoke" in result["amendment"]
    assert result["seeds"] == [290001]
    with (output / "sessions.csv").open() as stream:
        sessions = list(csv.DictReader(stream))
    assert len(sessions) == 7
    assert {r["strategy"] for r in sessions} == set(STRATEGIES)
    assert len(calls) == 7
    with pytest.raises(FileExistsError, match="never overwrite"):
        evaluate("research/study_plan_dqn_extended.json",
                 "models/selection.json", calibration_path, output)


# model and plan changes must invalidate the frozen decision
def test_frozen_winner_hash_guard(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    make_model().save(calibration_path)
    plan = "research/study_plan_dqn_extended.json"
    source = json.loads(open("models/selection.json").read())
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(source))
    check_frozen_inputs(plan, selected, calibration_path)
    source["model_files"]["extended_3"]["policy_sha256"] = "0" * 64
    selected.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="policy.pt"):
        check_frozen_inputs(plan, selected, calibration_path)


# editing the runner-up checkpoint hash must invalidate the whole experiment
def test_frozen_runner_up_hash_guard(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    make_model().save(calibration_path)
    plan = "research/study_plan_dqn_extended.json"
    source = json.loads(open("models/selection.json").read())
    selected = tmp_path / "selected.json"
    source["model_files"]["original_1"]["policy_sha256"] = "0" * 64
    selected.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Runner-up policy.pt"):
        check_frozen_inputs(plan, selected, calibration_path)
