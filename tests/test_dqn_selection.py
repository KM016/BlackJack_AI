"""Check model eligibility, independent evaluation and frozen selection rules."""

import copy
import json
from pathlib import Path

import pytest

from research.study_plan import load_plan
from rl.dqn import DQN
from rl.select import (CANDIDATE_NAMES, candidate_folders, choose_winner,
                       evaluate_session, summarize_candidate, validate_candidate)


# every candidate is drawn from exactly one of the two agreed training budgets
def test_candidate_folder_names():
    paths = candidate_folders("outputs/dqn", "outputs/dqn_extended")
    assert tuple(paths) == CANDIDATE_NAMES
    assert str(paths["original_2"]) == "outputs/dqn/train_2"
    assert str(paths["extended_3"]) == "outputs/dqn_extended/train_3"


# original and extended models must have the predeclared seed allocations
def test_saved_model_seed_and_reward_guards():
    plan = load_plan("research/study_plan.json")
    settings = {
        "seeds": plan["seeds"]["train_1"], "network_seed": 101,
        "initial_bankroll": "1000", "unit": "1", "rounds": 500,
        "epsilon_decay_steps": 10000, "environment_steps": 50000,
    }
    validate_candidate(settings, plan, "original_1")
    scaled = dict(settings, reward_scale=100.0)
    with pytest.raises(ValueError, match="reward-scaled"):
        validate_candidate(scaled, plan, "original_1")
    different_seeds = dict(settings, seeds=[999] * 100)
    with pytest.raises(ValueError, match="training seeds"):
        validate_candidate(different_seeds, plan, "original_1")


# selection uses mean log growth, not training profit or a single best session
def test_winner_rule_and_tie_breaks():
    candidates = [
        {"candidate": "original_1", "mean_log_growth": -0.01, "mean_max_drawdown": 0.08},
        {"candidate": "extended_2", "mean_log_growth": 0.003, "mean_max_drawdown": 0.10},
        {"candidate": "original_3", "mean_log_growth": 0.003, "mean_max_drawdown": 0.03},
    ]
    assert choose_winner(candidates)["candidate"] == "original_3"
    candidates[-1]["mean_max_drawdown"] = 0.10
    assert choose_winner(candidates)["candidate"] == "original_3"
    candidates[0]["mean_log_growth"] = 0.004
    assert choose_winner(candidates)["candidate"] == "original_1"


# a short fresh game checks greedy play, bankroll accounting and drawdown bounds
def test_real_environment_greedy_smoke():
    network = DQN().eval()
    result = evaluate_session(network, seed=987654321, name="original_1")
    assert result["rounds"] == 500
    assert result["bets"] + result["sits"] == 500
    assert 0 <= result["max_drawdown"] < 1
    assert result["final_bankroll"] > 0
    assert result["void_rounds"] >= 0


# result summaries must average whole episodes
def test_summary_is_per_session():
    rows = [
        {"log_growth": 0.1, "final_bankroll": 1100, "net_profit": 100,
         "max_drawdown": .10, "bets": 2, "rounds": 5},
        {"log_growth": -0.1, "final_bankroll": 900, "net_profit": -100,
         "max_drawdown": .20, "bets": 1, "rounds": 5},
    ]
    result = summarize_candidate(rows, "original_1")
    assert result["mean_log_growth"] == pytest.approx(0)
    assert result["betting_rate"] == pytest.approx(0.3)
    assert result["mean_max_drawdown"] == pytest.approx(0.15)
