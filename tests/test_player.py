import pytest
from decimal import Decimal
from blackjack.player import Player


# TEST 1: Check the starting bankroll and active bets
def test_initial_player():
    player = Player(1000)

    assert player.bankroll == Decimal("1000")
    assert player.bets == {}
    assert player.hands == []


# TEST 2: Check that placing a bet reduces the bankroll
def test_place_bet():
    player = Player(1000)

    player.place_bet("hand_1", 20)

    assert player.bankroll == 980
    assert player.bets["hand_1"] == 20


# TEST 3: Check that a normal win pays 1:1
def test_win():
    player = Player(1000)
    player.place_bet("hand_1", 20)

    profit = player.settle_bet("hand_1", "win")

    assert profit == 20
    assert player.bankroll == 1020
    assert player.bets == {}


# TEST 4: Check that a loss removes the wager
def test_loss():
    player = Player(1000)
    player.place_bet("hand_1", 20)

    profit = player.settle_bet("hand_1", "loss")

    assert profit == -20
    assert player.bankroll == 980


# TEST 5: Check that a push returns the original bet
def test_push():
    player = Player(1000)
    player.place_bet("hand_1", 20)

    profit = player.settle_bet("hand_1", "push")

    assert profit == 0
    assert player.bankroll == 1000


# TEST 6: Check that natural blackjack pays 3:2
def test_blackjack():
    player = Player(1000)
    player.place_bet("hand_1", 20)

    profit = player.settle_bet("hand_1", "blackjack")

    assert profit == 30
    assert player.bankroll == 1030


# TEST 7: Check that doubling down commits another stake
def test_double_down():
    player = Player(1000)
    player.place_bet("hand_1", 20)

    player.increase_bet("hand_1")

    assert player.bankroll == 960
    assert player.bets["hand_1"] == 40

    profit = player.settle_bet("hand_1", "win")

    assert profit == 40
    assert player.bankroll == 1040


# TEST 8: Check that multiple hands have separate bets
def test_multiple_hands():
    player = Player(1000)

    player.place_bet("hand_1", 20)
    player.place_bet("hand_2", 20)

    player.settle_bet("hand_1", "win")
    player.settle_bet("hand_2", "loss")

    assert player.bankroll == 1000
    assert player.bets == {}


# TEST 9: Check that the player cannot bet more than they have
def test_insufficient_bankroll():
    player = Player(100)

    with pytest.raises(ValueError):
        player.place_bet("hand_1", 200)

    assert player.bankroll == 100


# TEST 10: Check that invalid bets are rejected
def test_invalid_bets():
    player = Player(1000)

    with pytest.raises(ValueError):
        player.place_bet("hand_1", -10)

    with pytest.raises(ValueError):
        player.place_bet("hand_1", 0)

    with pytest.raises(ValueError):
        player.place_bet("hand_1", "NaN")


# TEST 11: Check that a bet cannot be placed twice on the same hand
def test_duplicate_bet():
    player = Player(1000)
    player.place_bet("hand_1", 20)

    with pytest.raises(ValueError):
        player.place_bet("hand_1", 20)

    assert player.bankroll == 980


# TEST 12: Check that an unaffordable double is rejected
def test_invalid_double():
    player = Player(30)
    player.place_bet("hand_1", 20)

    with pytest.raises(ValueError):
        player.increase_bet("hand_1")

    assert player.bankroll == 10
    assert player.bets["hand_1"] == 20


# TEST 13: Check that a hand cannot be settled twice
def test_duplicate_settlement():
    player = Player(1000)
    player.place_bet("hand_1", 20)
    player.settle_bet("hand_1", "win")

    with pytest.raises(ValueError):
        player.settle_bet("hand_1", "win")


# TEST 14: Check that invalid outcomes are rejected
def test_invalid_outcome():
    player = Player(1000)
    player.place_bet("hand_1", 20)

    with pytest.raises(ValueError):
        player.settle_bet("hand_1", "unknown")

    assert player.bankroll == 980
    assert player.bets["hand_1"] == 20


# TEST 15: Check that fractional wagers are handled exactly
def test_decimal_precision():
    player = Player("1000")
    player.place_bet("hand_1", "0.10")

    profit = player.settle_bet("hand_1", "blackjack")

    assert profit == Decimal("0.15")
    assert player.bankroll == Decimal("1000.15")


# TEST 16: Check that negative starting bankrolls are rejected
def test_invalid_bankroll():
    with pytest.raises(ValueError):
        Player(-100)

    with pytest.raises(ValueError):
        Player("NaN")
