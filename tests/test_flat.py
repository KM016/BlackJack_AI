from decimal import Decimal

import pytest

from blackjack.engine import BlackjackEngine
from blackjack.player import Player
from strategies.flat import FlatBetting


def betting_view(bankroll=1000, min_bet=1, max_bet=1000, can_bet=True, state="betting"):
    # make a simple public observation for testing betting decisions
    return {"state": state, "bankroll": bankroll, "min_bet": min_bet,
            "max_bet": max_bet, "can_bet": can_bet, "events": []}


# TEST 1: Check that flat betting always chooses the same stake
@pytest.mark.parametrize("result", ["win", "loss", "push", "blackjack"])
def test_fixed_stake(result):
    strategy = FlatBetting("10.50")
    observation = betting_view()
    observation["events"] = [{"type": "settlement", "outcome": result}]

    assert strategy.choose_wager(observation) == Decimal("10.50")


# TEST 2: Check that the stake is not reduced when the bankroll runs low
@pytest.mark.parametrize("bankroll,expected", [
    ("11", Decimal("10")), ("10", Decimal("10")), ("9.99", None), ("0", None)
])
def test_unaffordable_stake(bankroll, expected):
    strategy = FlatBetting(10)

    assert strategy.choose_wager(betting_view(bankroll=bankroll)) == expected


# TEST 3: Check that a fixed stake outside the table limits is rejected
@pytest.mark.parametrize("stake", ["0.5", "1001"])
def test_outside_table_limits(stake):
    strategy = FlatBetting(stake)

    with pytest.raises(ValueError):
        strategy.choose_wager(betting_view())


# TEST 4: Check that zero, negative and invalid stakes are rejected
@pytest.mark.parametrize("stake", [0, -1, True, False, 1.01, "NaN", "Infinity", "bad", None])
def test_invalid_stake(stake):
    with pytest.raises(ValueError):
        FlatBetting(stake)


# TEST 5: Check that the player sits out when betting is unavailable
def test_cannot_bet():
    strategy = FlatBetting(10)

    assert strategy.choose_wager(betting_view(can_bet=False)) is None


# TEST 6: Check that invalid public betting information is rejected
@pytest.mark.parametrize("changes", [
    {"bankroll": -1}, {"bankroll": 1.01}, {"bankroll": "NaN"},
    {"min_bet": 0}, {"max_bet": 0.5}, {"can_bet": 1},
    {"state": "player_turn"}
])
def test_invalid_betting_observation(changes):
    observation = betting_view()
    observation.update(changes)

    with pytest.raises(ValueError):
        FlatBetting(10).choose_wager(observation)


# TEST 7: Check that flat betting works with the real engine observation
def test_engine_observation():
    engine = BlackjackEngine(players={"p": Player(100)}, seed=42)

    assert FlatBetting(10).choose_wager(engine.betting_observation("p")) == Decimal("10")
