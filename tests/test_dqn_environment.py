import math
from decimal import Decimal

import pytest

from blackjack.engine import ShoeExhausted
from blackjack.rules import Rules
from rl.environment import BlackjackBettingEnv, FEATURE_NAMES, public_state
from strategies.hi_lo import HiLoCounter
from strategies.wager_grid import legal_wager_mask, wager_for_action


# TEST 1: The DQN receives exactly six public numbers and a six-action mask
def test_initial_observation():
    game = BlackjackBettingEnv(rounds=20)
    state = game.reset(230001)
    assert len(FEATURE_NAMES) == len(state["features"]) == len(state["legal_mask"]) == 6
    assert state["features"] == (0., 0., 1., 1., 0., 1.)
    assert state["legal_mask"] == (True,) * 6
    assert set(state) == {"features", "legal_mask"}

    # changing a returned observation must never change the game's real state
    state["features"] = (100,) * 6
    assert game.get_observation()["features"] == (0., 0., 1., 1., 0., 1.)


# TEST 2: Invalid actions must not deal cards or change the bankroll
def test_invalid_actions_do_not_advance():
    for action in (-1, 6, 1.0, True, None, "1"):
        game = BlackjackBettingEnv()
        game.reset(230001)
        with pytest.raises(ValueError):
            game.step(action)
        assert game._engine.round_number == 0
        assert game._engine.players["agent"].bankroll == 1000


# TEST 3: The agent may sit out while the reference still plays and reveals cards
def test_sitting_out_still_advances_the_shoe():
    game = BlackjackBettingEnv(rounds=5)
    game.reset(230001)
    initial_money = game._engine.players["agent"].bankroll
    state, reward, done, info = game.step(0)
    assert reward == 0
    assert not done
    assert info["wager"] is None
    assert info["net_profit"] == 0
    assert game._engine.round_number == 1
    assert game._engine.players["agent"].bankroll == initial_money
    assert game.counter.cards_seen >= 3
    assert state["features"][2] < 1
    assert len(game._engine.players["reference"].hands) >= 1
    assert game._engine.players["agent"].hands == []


# TEST 4: Positive wagers change bankroll and deliver the exact log reward
def test_real_bet_and_log_reward():
    game = BlackjackBettingEnv(rounds=5)
    game.reset(230001)
    next_state, reward, done, info = game.step(1)
    assert info["wager"] == 1
    assert info["bankroll_after"] - info["bankroll_before"] == info["net_profit"]
    assert reward == pytest.approx(math.log(float(info["bankroll_after"] / info["bankroll_before"])))
    assert info["round_status"] == "complete"
    assert not done
    assert next_state["features"][5] == pytest.approx(.8)


# TEST 5: Risk masks are shared with other strategies, including fractional unit sizes
def test_risk_mask():
    examples = (
        ("49.99", (True, False, False, False, False, False)),
        ("100", (True, True, True, False, False, False)),
        ("200", (True, True, True, True, False, False)),
        ("400", (True, True, True, True, True, True)),
    )
    for money, expected in examples:
        game = BlackjackBettingEnv(initial_bankroll=money)
        state = game.reset(230001)
        if Decimal(money) < 50:
            assert game.done
            assert state["legal_mask"] == (False,) * 6
        else:
            assert state["legal_mask"] == expected
            assert legal_wager_mask(game._engine.betting_observation("agent")) == expected


def test_configurable_units_use_the_same_grid():
    game = BlackjackBettingEnv(unit="2", rounds=2)
    game.reset(230001)
    assert wager_for_action(2, 2) == 4
    _, _, _, info = game.step(2)
    assert info["wager"] == Decimal(4)
    assert game._engine.players["reference"].hands[0].hand is not None
    assert wager_for_action(0, 2) is None


# TEST 6: Don't allow unsafe wagers or advance after a terminal episode
def test_unsafe_wager_and_terminal_state():
    game = BlackjackBettingEnv(initial_bankroll=100, rounds=1)
    state = game.reset(230001)
    assert state["legal_mask"] == (True, True, True, False, False, False)
    with pytest.raises(ValueError):
        game.step(5)
    assert game._engine.round_number == 0
    state, _, done, info = game.step(0)
    assert done and info["terminal_reason"] == "horizon"
    assert state["legal_mask"] == (False,) * 6
    with pytest.raises(ValueError):
        game.step(0)


def test_terminal_when_no_positive_stake_is_affordable():
    game = BlackjackBettingEnv(initial_bankroll="50.5")
    game.reset(230001)
    # this seeded round loses one unit, leaving <50 units to fund a minimum bet
    _, _, done, info = game.step(1)
    assert done
    assert info["terminal_reason"] == "no_positive_wager"
    assert info["bankroll_after"] < 50


# TEST 7: Rewards telescope to the total log change in bankroll
def test_sum_of_log_rewards_matches_final_log_bankroll():
    game = BlackjackBettingEnv(rounds=75)
    state = game.reset(230001)
    rewards = []
    while not game.done:
        action = 1 if state["legal_mask"][1] else 0
        state, reward, _, _ = game.step(action)
        rewards.append(reward)
    final_money = game._engine.players["agent"].bankroll
    assert len(rewards) == 75
    assert sum(rewards) == pytest.approx(math.log(float(final_money / 1000)), abs=1e-12)


# TEST 8: Fixed seeds produce exactly reproducible sessions
def test_episode_reproducibility():
    outcomes = []
    for _ in range(2):
        game = BlackjackBettingEnv(rounds=30)
        state = game.reset(230001)
        record = []
        while not game.done:
            action = 2 if state["legal_mask"][2] else 0
            state, reward, done, info = game.step(action)
            record.append((state, reward, done, info))
        outcomes.append(record)
    assert outcomes[0] == outcomes[1]


# TEST 9: Public state resets the effective count and remaining cards on reshuffle
def test_public_shuffle_features():
    game = BlackjackBettingEnv(rounds=150)
    state = game.reset(230001)
    shuffled = False
    while not game.done:
        state, _, _, _ = game.step(0)
        next_view = game._engine.betting_observation("agent")
        if next_view["reshuffle_pending"] and not game.done:
            assert state["features"][0] == 0
            assert state["features"][1] == 0
            assert state["features"][2] == 1
            shuffled = True
    assert shuffled


# TEST 10: Bad setup and seeds must be rejected
def test_invalid_setup():
    examples = (
        {"initial_bankroll": 0}, {"unit": 0}, {"rounds": 0}, {"rounds": True},
        {"rules": Rules(num_decks=2)}, {"unit": 2000},
    )
    for kwargs in examples:
        with pytest.raises(ValueError):
            BlackjackBettingEnv(**kwargs)


def test_invalid_seed_and_reset_required():
    game = BlackjackBettingEnv()
    with pytest.raises(ValueError):
        game.get_observation()
    with pytest.raises(ValueError):
        game.step(0)
    for seed in (-1, 1.5, True, "10"):
        with pytest.raises(ValueError):
            game.reset(seed)


# TEST 11: A voided round has no profit and no reward
def test_void_round_refunds_and_uses_zero_reward():
    game = BlackjackBettingEnv(rounds=2)
    game.reset(230001)

    # simulate the compulsory dealer draw running out of cards
    def void_dealer_round():
        game._engine._void_round()
        raise ShoeExhausted("Simulated compulsory draw shortage")

    game._engine._play_dealer = void_dealer_round
    _, reward, done, info = game.step(1)
    assert info["round_status"] == "void"
    assert info["net_profit"] == 0
    assert info["bankroll_after"] == info["bankroll_before"]
    assert reward == 0
    assert not done
