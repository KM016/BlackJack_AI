import pydealer
from blackjack.dealer import Dealer


def card(value):
    # create a PyDealer card for testing
    return pydealer.Card(value, "Spades")


# TEST 1: Check that a new dealer starts with an empty hand
def test_initial_dealer():
    dealer = Dealer()

    assert dealer.hand.cards == []
    assert dealer.revealed is False
    assert dealer.get_upcard() is None


# TEST 2: Check that the dealer's first card is the upcard
def test_upcard():
    dealer = Dealer([card("Ace"), card("7")])

    assert dealer.get_upcard().value == "Ace"
    assert dealer.revealed is False


# TEST 3: Check that an Ace and King produce natural blackjack
def test_blackjack():
    dealer = Dealer([card("Ace"), card("King")])

    assert dealer.peek() is True
    assert dealer.revealed is True


# TEST 4: Check that blackjack works with a ten-value upcard
def test_ten_upcard_blackjack():
    dealer = Dealer([card("Queen"), card("Ace")])

    assert dealer.peek() is True
    assert dealer.revealed is True


# TEST 5: Check that an unsuccessful peek keeps the hole card hidden
def test_failed_peek():
    dealer = Dealer([card("Ace"), card("7")])

    assert dealer.peek() is False
    assert dealer.revealed is False


# TEST 6: Check that the dealer cannot have blackjack with a 9 upcard
def test_no_blackjack_peek():
    dealer = Dealer([card("9"), card("Ace")])

    assert dealer.peek() is False
    assert dealer.revealed is False


# TEST 7: Check that the hole card can be revealed manually
def test_reveal():
    dealer = Dealer([card("9"), card("7")])

    dealer.reveal()

    assert dealer.revealed is True


# TEST 8: Check that the dealer must hit below 17
def test_dealer_hit():
    dealer = Dealer([card("9"), card("7")])

    assert dealer.hand.get_total() == 16
    assert dealer.should_hit() is True


# TEST 9: Check that the dealer stands on hard 17
def test_stand_hard_17():
    dealer = Dealer([card("10"), card("7")])

    assert dealer.should_hit() is False


# TEST 10: Check that the dealer stands on soft 17
def test_stand_soft_17():
    dealer = Dealer([card("Ace"), card("6")])

    assert dealer.hand.is_soft() is True
    assert dealer.should_hit() is False


# TEST 11: Check that the dealer draws cards until reaching 17
def test_dealer_play():
    dealer = Dealer([card("8"), card("6")])

    # supply a 2 followed by an Ace
    cards = iter([card("2"), card("Ace")])

    dealer.play(lambda: next(cards))

    assert dealer.hand.get_total() == 17
    assert len(dealer.hand.cards) == 4
    assert dealer.revealed is True


# TEST 12: Check that the dealer does not draw when already on 17
def test_no_extra_cards():
    dealer = Dealer([card("10"), card("7")])

    # raise an error if the dealer incorrectly tries to draw
    def draw():
        raise AssertionError("Dealer should not draw")

    dealer.play(draw)

    assert len(dealer.hand.cards) == 2
    assert dealer.revealed is True


# TEST 13: Check that the dealer stops drawing after going bust
def test_dealer_bust():
    dealer = Dealer([card("10"), card("6")])

    # drawing a King makes the dealer bust with 26
    dealer.play(lambda: card("King"))

    assert dealer.hand.get_total() == 26
    assert dealer.hand.is_bust() is True
    assert len(dealer.hand.cards) == 3
