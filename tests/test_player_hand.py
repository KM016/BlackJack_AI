import pydealer
import pytest
from blackjack.player_hand import PlayerHand


def card(value):
    # create a PyDealer card for testing
    return pydealer.Card(value, "Spades")


# TEST 1: Check that a new player hand is initialised correctly
def test_initial_hand():
    hand = PlayerHand([card("8"), card("7")], hand_id="hand_1")

    assert hand.hand.get_total() == 15
    assert hand.hand_id == "hand_1"
    assert hand.has_hit is False
    assert hand.doubled is False
    assert hand.finished is False
    assert hand.outcome is None
    assert hand.net_profit is None


# TEST 2: Check that an empty hand can receive its initial cards
def test_empty_hand():
    hand = PlayerHand()

    assert hand.hand.cards == []
    assert hand.finished is False

    hand.hand.add_card(card("8"))
    hand.hand.add_card(card("7"))

    assert hand.hand.get_total() == 15


# TEST 3: Check that hitting adds a card
def test_hit():
    hand = PlayerHand([card("8"), card("7")])

    hand.hit(card("4"))

    assert hand.hand.get_total() == 19
    assert hand.has_hit is True
    assert hand.finished is False


# TEST 4: Check that standing finishes the hand
def test_stand():
    hand = PlayerHand([card("8"), card("7")])

    hand.stand()

    assert hand.finished is True
    assert hand.has_hit is False


# TEST 5: Check that a player cannot hit after standing
def test_hit_after_stand():
    hand = PlayerHand([card("8"), card("7")])
    hand.stand()

    with pytest.raises(ValueError):
        hand.hit(card("2"))

    assert hand.hand.get_total() == 15


# TEST 6: Check that a player cannot stand twice
def test_double_stand():
    hand = PlayerHand([card("8"), card("7")])
    hand.stand()

    with pytest.raises(ValueError):
        hand.stand()


# TEST 7: Check that going bust automatically finishes the hand
def test_bust():
    hand = PlayerHand([card("King"), card("8")])

    hand.hit(card("5"))

    assert hand.hand.get_total() == 23
    assert hand.hand.is_bust() is True
    assert hand.finished is True


# TEST 8: Check that reaching 21 automatically finishes the hand
def test_hit_to_21():
    hand = PlayerHand([card("5"), card("6")])

    hand.hit(card("King"))

    assert hand.hand.get_total() == 21
    assert hand.finished is True


# TEST 9: Check that natural blackjack starts as a finished hand
def test_natural_blackjack():
    hand = PlayerHand([card("Ace"), card("King")])

    assert hand.hand.is_blackjack() is True
    assert hand.finished is True


# TEST 10: Check that doubling down adds one card and ends the turn
def test_double_down():
    hand = PlayerHand([card("8"), card("7")])

    assert hand.can_double() is True

    hand.double_down(card("2"))

    assert hand.hand.get_total() == 17
    assert len(hand.hand.cards) == 3
    assert hand.has_hit is True
    assert hand.doubled is True
    assert hand.finished is True


# TEST 11: Check that doubling is not allowed after hitting
def test_double_after_hit():
    hand = PlayerHand([card("5"), card("6")])

    hand.hit(card("2"))

    assert hand.can_double() is False

    with pytest.raises(ValueError):
        hand.double_down(card("3"))


# TEST 12: Check that matching card ranks can be split
def test_can_split():
    hand = PlayerHand([card("8"), card("8")])

    assert hand.can_split() is True
    assert hand.can_double() is True


# TEST 13: Check that different ten-value cards cannot be split
def test_different_ten_values():
    hand = PlayerHand([card("King"), card("Queen")])

    assert hand.hand.get_total() == 20
    assert hand.can_split() is False


# TEST 14: Check that splitting is not allowed after hitting
def test_split_after_hit():
    hand = PlayerHand([card("5"), card("5")])

    hand.hit(card("2"))

    assert hand.can_split() is False


# TEST 15: Check that a split hand waits for its replacement card
def test_waiting_split_hand():
    hand = PlayerHand([card("8")], from_split=True)

    assert hand.awaiting_card is True
    assert hand.finished is False
    assert hand.can_double() is False
    assert hand.can_split() is False

    with pytest.raises(ValueError):
        hand.hit(card("2"))

    with pytest.raises(ValueError):
        hand.stand()


# TEST 16: Check that a split hand can play after receiving a card
def test_receive_split_card():
    hand = PlayerHand([card("8")], from_split=True)

    hand.receive_split_card(card("3"))

    assert hand.awaiting_card is False
    assert hand.hand.get_total() == 11
    assert hand.finished is False
    assert hand.has_hit is False
    assert hand.can_double() is True


# TEST 17: Check that non-Ace split hands can be split again
def test_resplit():
    hand = PlayerHand([card("8")], from_split=True)

    hand.receive_split_card(card("8"))

    assert hand.can_split() is True


# TEST 18: Check that split Aces finish after one replacement card
def test_split_aces():
    hand = PlayerHand(
        [card("Ace")],
        from_split=True,
        split_aces=True
    )

    hand.receive_split_card(card("9"))

    assert hand.hand.get_total() == 20
    assert hand.finished is True
    assert hand.can_double() is False
    assert hand.can_split() is False

    with pytest.raises(ValueError):
        hand.hit(card("2"))


# TEST 19: Check that a split Ace and King is not natural blackjack
def test_split_ace_21():
    hand = PlayerHand(
        [card("Ace")],
        from_split=True,
        split_aces=True
    )

    hand.receive_split_card(card("King"))

    assert hand.hand.get_total() == 21
    assert hand.hand.is_blackjack() is False
    assert hand.finished is True


# TEST 20: Check that a split hand cannot receive two replacement cards
def test_duplicate_split_card():
    hand = PlayerHand([card("8")], from_split=True)

    hand.receive_split_card(card("3"))

    with pytest.raises(ValueError):
        hand.receive_split_card(card("4"))

    assert len(hand.hand.cards) == 2
