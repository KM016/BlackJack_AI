import pydealer
from blackjack.hand import Hand


def card(value):
    # create a PyDealer card for testing
    return pydealer.Card(value, "Spades")


# TEST 1: Check that a new hand starts empty
def test_empty_hand():
    hand = Hand()

    assert hand.cards == []
    assert hand.get_total() == 0
    assert hand.is_soft() is False
    assert hand.is_blackjack() is False
    assert hand.is_bust() is False
    assert hand.from_split is False


# TEST 2: Check that a hand stores the cards we give it
def test_initial_cards():
    cards = [card("8"), card("7")]
    hand = Hand(cards)

    assert len(hand.cards) == 2
    assert hand.get_total() == 15

    # changing the original list should not change the hand
    cards.append(card("5"))
    assert len(hand.cards) == 2


# TEST 3: Check that adding a card updates the hand
def test_add_card():
    hand = Hand([card("8"), card("7")])

    hand.add_card(card("5"))

    assert len(hand.cards) == 3
    assert hand.get_total() == 20


# TEST 4: Check that numbered cards and face cards have the correct values
def test_card_values():
    hand = Hand([card("King"), card("Queen")])
    assert hand.get_total() == 20

    hand = Hand([card("Jack"), card("5")])
    assert hand.get_total() == 15


# TEST 5: Check that an Ace counts as 11 when possible
def test_soft_hand():
    hand = Hand([card("Ace"), card("6")])

    assert hand.get_total() == 17
    assert hand.is_soft() is True


# TEST 6: Check that an Ace changes from 11 to 1 to avoid busting
def test_ace_conversion():
    hand = Hand([card("Ace"), card("6")])

    hand.add_card(card("10"))

    assert hand.get_total() == 17
    assert hand.is_soft() is False
    assert hand.is_bust() is False


# TEST 7: Check that multiple Aces are counted correctly
def test_multiple_aces():
    hand = Hand([card("Ace"), card("Ace"), card("9")])

    # one Ace is worth 11 and the other is worth 1
    assert hand.get_total() == 21
    assert hand.is_soft() is True


# TEST 8: Check that an Ace and a ten-value card make blackjack
def test_blackjack():
    hand = Hand([card("Ace"), card("King")])

    assert hand.get_total() == 21
    assert hand.is_blackjack() is True


# TEST 9: Check that reaching 21 with three cards is not blackjack
def test_non_natural_21():
    hand = Hand([card("7"), card("7"), card("7")])

    assert hand.get_total() == 21
    assert hand.is_blackjack() is False


# TEST 10: Check that a split hand cannot have natural blackjack
def test_split_hand():
    hand = Hand([card("Ace"), card("King")], from_split=True)

    assert hand.get_total() == 21
    assert hand.from_split is True
    assert hand.is_blackjack() is False


# TEST 11: Check that a hand exceeding 21 is bust
def test_bust():
    hand = Hand([card("King"), card("9"), card("5")])

    assert hand.get_total() == 24
    assert hand.is_bust() is True
