import json
import math
import sys
from pathlib import Path

import pytest
import torch

from rl.dqn import ReplayBuffer
from rl.train import main, train


# check that only replay rewards are multiplied while real bankroll returns stay unchanged
def test_scaled_training_keeps_real_financial_results(tmp_path, monkeypatch):
    replay_rewards = []
    original_add = ReplayBuffer.add

    # record the rewards passed to the replay buffer during a short real simulation
    def record_add(self, state, action, reward, next_state, done):
        replay_rewards.append(reward)
        original_add(self, state, action, reward, next_state, done)

    monkeypatch.setattr(ReplayBuffer, "add", record_add)
    folder = tmp_path / "scaled"
    network, log = train([991001], folder, network_seed=405, rounds=70, reward_scale=100.0)

    # the buffer sees 100 times the original log-bankroll rewards
    assert len(replay_rewards) == 70
    assert any(reward != 0 for reward in replay_rewards)
    assert sum(replay_rewards) / 100 == pytest.approx(log[0]["log_growth"])

    # the reporting still matches the true change in the simulated bankroll
    assert log[0]["log_growth"] == pytest.approx(math.log(log[0]["final_bankroll"] / 1000))
    assert log[0]["updates"] > 0
    assert all(torch.isfinite(value).all() for value in network.parameters())
    assert json.loads((folder / "settings.json").read_text())["reward_scale"] == 100.0


# setting the multiplier to one must reproduce the unscaled model and log exactly
def test_unit_reward_scaling_preserves_original_training(tmp_path):
    baseline, baseline_log = train([991101], tmp_path / "baseline", network_seed=719, rounds=70)
    unit, unit_log = train([991101], tmp_path / "unit", network_seed=719, rounds=70, reward_scale=1.0)

    assert baseline_log == unit_log
    assert all(torch.equal(a, b) for a, b in zip(baseline.parameters(), unit.parameters()))
    assert json.loads((tmp_path / "baseline" / "settings.json").read_text())["reward_scale"] == 1.0


# reject invalid learning multipliers before creating any output files
def test_reward_scale_must_be_positive_and_finite(tmp_path):
    for scale in (0, -2, float("inf"), -float("inf"), float("nan"), True, "100", None):
        with pytest.raises(ValueError, match="Reward scale"):
            train([991201], tmp_path / "invalid", network_seed=101, rounds=1, reward_scale=scale)
    assert not (tmp_path / "invalid").exists()


# confirm the new command-line flag actually reaches the training function
def test_reward_scale_cli_flag(tmp_path, monkeypatch):
    folder = tmp_path / "command_line"
    plan = json.loads(Path("research/study_plan.json").read_text())
    # Exercise the CLI with a fresh plan, without replaying research seed roles.
    plan["seeds"] = {role: [992001 + index]
                     for index, role in enumerate(plan["seeds"])}
    plan_path = tmp_path / "test_plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(sys, "argv", [
        "rl.train", "--plan", str(plan_path), "--run", "train_1",
        "--output", str(folder), "--network-seed", "991", "--episodes", "1",
        "--rounds", "2", "--reward-scale", "100",
    ])
    main()

    settings = json.loads((folder / "settings.json").read_text())
    assert settings["reward_scale"] == 100.0
    assert settings["seeds"] == plan["seeds"]["train_1"]
    assert settings["environment_steps"] == 2
