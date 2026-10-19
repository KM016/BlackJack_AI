import csv
import json
import math
import random
from copy import deepcopy

import pytest
import torch

from rl.dqn import DQN, ReplayBuffer, calculate_targets, choose_action, train_batch
from rl.environment import BlackjackBettingEnv, FEATURE_NAMES
from rl.policy import DQNBetting, load_policy, save_policy
from rl.train import exploration_rate, train
from strategies.wager_grid import wager_for_action


# check that the network uses our original six-input, six-output architecture
def test_network_shape():
    network = DQN()
    assert network(torch.zeros(4, 6)).shape == (4, 6)
    assert [(layer.in_features, layer.out_features) for layer in network.layers if isinstance(layer, torch.nn.Linear)] == [
        (6, 64), (64, 64), (64, 6)
    ]


# check that random exploration and greedy decisions both respect action masks
def test_choose_action_never_uses_illegal_wagers():
    network = DQN()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.layers[-1].bias.copy_(torch.tensor([1., 2., 30., 40., 50., 60.]))
    observation = {"features": (0,) * 6, "legal_mask": (True, True, False, False, False, False)}
    assert choose_action(network, observation, 0) == 1
    rng = random.Random(13)
    assert {choose_action(network, observation, 1, rng) for _ in range(100)} == {0, 1}
    with pytest.raises(ValueError):
        choose_action(network, {"features": (0,) * 6, "legal_mask": (False,) * 6})
    with pytest.raises(ValueError):
        choose_action(network, observation, -0.2)


# check that next-state Q targets ignore both illegal actions and terminal futures
def test_targets_apply_next_mask_and_done():
    target = DQN()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        target.layers[-1].bias.copy_(torch.tensor([1., 2., 30., 40., 50., 60.]))
    rewards = torch.tensor([0.5, -1.0])
    features = torch.zeros(2, 6)
    done = torch.tensor([False, True])
    masks = torch.tensor([[True, True, False, False, False, False], [False] * 6])
    values = calculate_targets(target, rewards, features, done, masks)
    assert values.tolist() == pytest.approx([2.5, -1.0])
    with pytest.raises(ValueError, match="nonterminal"):
        calculate_targets(target, rewards, features, torch.tensor([False, False]), masks)


# remember the latest fixed number of rounds, without exposing a mutable game state
def test_replay_buffer_capacity_and_sampling():
    memory = ReplayBuffer(2)
    state = {"features": [0] * 6, "legal_mask": [True] * 6}
    next_state = deepcopy(state)
    for action in range(3):
        memory.add(state, action, action / 10, next_state, action == 2)
    state["features"][0] = 100
    assert len(memory) == 2
    sample = memory.sample(2, random.Random(1))
    assert {row[1] for row in sample} == {1, 2}
    assert all(row[0][0] == 0 for row in sample)
    with pytest.raises(ValueError):
        ReplayBuffer(0)


# a minibatch of experiences must really change the network's learned weights
def test_training_batch_changes_weights():
    torch.manual_seed(30)
    online = DQN()
    target = DQN()
    target.load_state_dict(online.state_dict())
    optimiser = torch.optim.Adam(online.parameters(), lr=0.001)
    before = [value.clone() for value in online.parameters()]
    batch = [((0.,) * 6, 1, 1.0, (0.,) * 6, True, (False,) * 6) for _ in range(64)]
    loss = train_batch(online, target, optimiser, batch)
    assert loss > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, online.parameters()))


# repeated synthetic rewards teach the DQN to prefer the better legal action
def test_dqn_learns_a_toy_advantage():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    online = DQN()
    target = DQN()
    target.load_state_dict(online.state_dict())
    optimiser = torch.optim.Adam(online.parameters(), lr=0.005)
    state = (0.,) * 6
    mask = (True, True, False, False, False, False)
    for _ in range(60):
        batch = [
            (state, 0, 0.0, state, True, (False,) * 6),
            (state, 1, 1.0, state, True, (False,) * 6),
        ] * 32
        train_batch(online, target, optimiser, batch)
    assert choose_action(online, {"features": state, "legal_mask": mask}, 0) == 1


# check the original exploration schedule: full exploration then five percent
def test_exploration_schedule():
    assert exploration_rate(0) == 1.0
    assert exploration_rate(5000) == pytest.approx(0.525)
    assert exploration_rate(10000) == pytest.approx(0.05)
    assert exploration_rate(12000) == pytest.approx(0.05)


# saved model weights and settings must reproduce the same real betting decisions
def test_save_and_load_policy(tmp_path):
    settings = {
        "version": "blackjack_dqn_v1", "features": list(FEATURE_NAMES),
        "wagers": [0, 1, 2, 4, 6, 8], "num_decks": 6,
        "rounds": 12, "initial_bankroll": "1000", "unit": "1",
    }
    torch.manual_seed(8)
    network = DQN()
    save_policy(network, settings, tmp_path / "model")
    first = load_policy(tmp_path / "model")
    second = load_policy(tmp_path / "model")
    game = BlackjackBettingEnv(rounds=12)
    game.reset(999001)
    first.reset()
    second.reset()
    for _ in range(12):
        view = game._engine.betting_observation("agent")
        first_wager = first.choose_wager(view)
        second_wager = second.choose_wager(view)
        assert first_wager == second_wager
        assert first_wager == wager_for_action(first.last_decision["action_index"])
        assert first.last_decision == second.last_decision
        game.step(first.last_decision["action_index"])
    with pytest.raises(FileExistsError):
        save_policy(network, settings, tmp_path / "model")


# run tiny training sessions and save the policy without using final research seeds
def test_training_smoke_and_seed_reproducibility(tmp_path):
    results = []
    for folder in ("first", "second"):
        network, log = train([999111, 999112], tmp_path / folder, network_seed=913, rounds=8)
        results.append(log)
        assert len(log) == 2
        assert (tmp_path / folder / "policy.pt").exists()
        settings = json.loads((tmp_path / folder / "settings.json").read_text())
        assert settings["seeds"] == [999111, 999112]
        assert settings["rounds"] == 8
        assert settings["network_seed"] == 913
        with (tmp_path / folder / "training_log.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 2
        policy = load_policy(tmp_path / folder)
        assert isinstance(policy, DQNBetting)
        game = BlackjackBettingEnv(rounds=8)
        game.reset(999222)
        policy.reset()
        while not game.done:
            observation = game._engine.betting_observation("agent")
            policy.choose_wager(observation)
            game.step(policy.last_decision["action_index"])
        assert game.steps == 8
    assert results[0] == results[1]
    with pytest.raises(FileExistsError):
        train([999111], tmp_path / "first", network_seed=913, rounds=8)
    with pytest.raises(ValueError):
        train([999111, 999111], tmp_path / "broken", network_seed=913, rounds=8)
