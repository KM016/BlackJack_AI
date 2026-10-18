import pytest
from decimal import Decimal
from dataclasses import FrozenInstanceError
from blackjack.rules import Rules, money


# TEST 1: Check that money is converted to Decimal
def test_money():
    assert money("10.50") == Decimal("10.50")
    assert money(20) == Decimal("20")


# TEST 2: Check that invalid monetary values are rejected
def test_invalid_money():
    for value in ("NaN", "Infinity", "abc"):
        with pytest.raises(ValueError):
            money(value)


# TEST 3: Check the default blackjack rules
def test_default_rules():
    rules = Rules()

    assert rules.num_decks == 6
    assert rules.penetration == 0.75
    assert rules.min_bet == Decimal("1")
    assert rules.max_bet == Decimal("1000")
    assert rules.max_split_hands == 4


# TEST 4: Check that valid custom rules are accepted
def test_custom_rules():
    rules = Rules(num_decks=2, min_bet="5", max_bet="500")

    assert rules.num_decks == 2
    assert rules.min_bet == Decimal("5")
    assert rules.max_bet == Decimal("500")


# TEST 5: Check that invalid rules are rejected
def test_invalid_rules():
    with pytest.raises(ValueError):
        Rules(num_decks=0)

    with pytest.raises(ValueError):
        Rules(penetration=1.5)

    with pytest.raises(ValueError):
        Rules(max_split_hands=5)

    with pytest.raises(ValueError):
        Rules(min_bet=100, max_bet=50)


# TEST 6: Check that casino rules cannot change after creation
def test_frozen_rules():
    rules = Rules()

    with pytest.raises(FrozenInstanceError):
        rules.num_decks = 2
