from decimal import Decimal
from itertools import product

import pytest

from blackjack.engine import BlackjackEngine
from blackjack.player import Player
from strategies.basic import BasicStrategy
from strategies.flat import FlatBetting
from strategies.hi_lo import HiLoBetting
from strategies.runner import run_round
from strategies.wager_grid import (DiscreteBetting, legal_wager_mask,
                                   proposed_action, wager_for_action)


# make a public betting observation for testing the common wager limits
def make_view(**changes):
    observation = {"state": "betting", "round_number": 0, "num_decks": 6,
                   "bankroll": Decimal(1000), "min_bet": Decimal(1),
                   "max_bet": Decimal(1000), "can_bet": True,
                   "reshuffle_pending": False, "events": []}
    observation.update(changes)
    return observation


# TEST 1: Check that the six betting actions have the original wager sizes
@pytest.mark.parametrize("action,expected", [(0, None), (1, 1), (2, 2),
                                               (3, 4), (4, 6), (5, 8)])
def test_wager_action(action, expected):
    assert wager_for_action(action) == expected


# TEST 2: Check invalid betting action indices are rejected
@pytest.mark.parametrize("action", [-1, 6, 1.0, True, "1"])
def test_invalid_action(action):
    with pytest.raises(ValueError):
        wager_for_action(action)


# TEST 3: Check the two-percent initial wager limit
@pytest.mark.parametrize("bankroll,expected", [
    (Decimal("49.99"), (True, False, False, False, False, False)),
    (Decimal(100), (True, True, True, False, False, False)),
    (Decimal(200), (True, True, True, True, False, False)),
    (Decimal(300), (True, True, True, True, True, False)),
    (Decimal(400), (True, True, True, True, True, True)),
])
def test_bankroll_cap(bankroll, expected):
    assert legal_wager_mask(make_view(bankroll=bankroll)) == expected


# TEST 4: Check table min/max limits and whether the player can bet
@pytest.mark.parametrize("changes,expected", [
    ({"min_bet": Decimal(3)}, (True, False, False, True, True, True)),
    ({"max_bet": Decimal(5)}, (True, True, True, True, False, False)),
    ({"can_bet": False}, (True, False, False, False, False, False)),
])
def test_table_limits(changes, expected):
    assert legal_wager_mask(make_view(**changes)) == expected


# TEST 5: Check every proposal rounds down to the biggest legal wager
@pytest.mark.parametrize("proposed,expected", [
    (None, 0), ("-5", 0), ("0.99", 0), (1, 1), ("1.99", 1),
    ("2", 2), ("3.999", 2), (4, 3), (5, 3), (6, 4), (8, 5), (99, 5)
])
def test_round_down(proposed, expected):
    assert proposed_action(proposed, (True,) * 6) == expected


# TEST 6: Check adaptation never chooses an illegal action
@pytest.mark.parametrize("proposal", [None, 0, 1, 2, 4, 6, 8, 10])
def test_all_possible_masks(proposal):
    for suffix in product((False, True), repeat=5):
        mask = (True,) + suffix
        selected = proposed_action(proposal, mask)
        assert mask[selected]
        assert selected == max([0] + [index for index in range(1, 6)
                                      if mask[index] and proposal is not None
                                      and wager_for_action(index) <= proposal])


# TEST 7: Check invalid masks and proposals are rejected
@pytest.mark.parametrize("mask,proposal", [
    ((False,) * 6, 1), ((True,), 1), ((True, True, True, 1, True, True), 1),
    ((True,) * 6, True), ((True,) * 6, 1.0), ((True,) * 6, "NaN")
])
def test_invalid_proposal(mask, proposal):
    with pytest.raises(ValueError):
        proposed_action(proposal, mask)


# TEST 8: Check Flat Betting follows the same bankroll limit
@pytest.mark.parametrize("bankroll,expected", [(100, 2), (200, 4), (400, 8)])
def test_wrapped_flat(bankroll, expected):
    policy = DiscreteBetting(FlatBetting(10))
    assert policy.choose_wager(make_view(bankroll=Decimal(bankroll))) == expected


# TEST 9: Check Hi-Lo and Flat Betting are both wrapped without changing their rules
def test_wrapped_hi_lo():
    policy = DiscreteBetting(HiLoBetting())
    assert policy.choose_wager(make_view()) == 1
    assert policy.last_action == 1


# TEST 10: Check Never Bet always sits out without consuming cards
def test_never_bet():
    engine = BlackjackEngine({"p": Player(1000)}, seed=42)
    policy = DiscreteBetting()
    before = list(engine.shoe.shoe.cards)
    result = run_round(engine, {"p": BasicStrategy()}, {"p": policy})
    assert result["status"] == "no_active_seats"
    assert engine.round_number == 0
    assert before == list(engine.shoe.shoe.cards)
    assert policy.last_action == 0


# TEST 11: Check resets clear the card count between independent sessions
def test_reset_count():
    policy = DiscreteBetting(HiLoBetting())
    policy.policy.counter.running_count = 12
    policy.choose_wager(make_view())
    policy.reset()
    assert policy.policy.counter.running_count == 0
    assert policy.last_action is None


# TEST 12: Check no sub-minimum bet is rounded up into a wager
def test_no_forced_bet():
    policy = DiscreteBetting(FlatBetting("0.5"))
    assert policy.choose_wager(make_view(min_bet=Decimal("0.5"))) is None
