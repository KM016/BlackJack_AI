from collections import Counter

import pytest
from blackjack.shoe import Shoe


# TEST 1: Check that a new six-deck shoe contains 312 cards
def test_initial_shoe():
    shoe = Shoe(seed=42)

    assert shoe.num_decks == 6
    assert shoe.get_remaining() == 312
    assert shoe.get_penetration() == 0
    assert shoe.needs_reshuffle() is False
    assert shoe.shuffle_count == 1


# TEST 2: Check that dealing cards removes them from the shoe
def test_dealing_cards():
    shoe = Shoe(seed=42)

    cards = shoe.deal(2)

    assert len(cards) == 2
    assert shoe.get_remaining() == 310


# TEST 3: Check that dealing one card works by default
def test_default_deal():
    shoe = Shoe(seed=42)

    cards = shoe.deal()

    assert len(cards) == 1
    assert shoe.get_remaining() == 311


# TEST 4: Check that penetration is calculated correctly
def test_penetration():
    shoe = Shoe(seed=42)

    # deal half the cards
    shoe.deal(156)

    assert shoe.get_penetration() == 0.5
    assert shoe.needs_reshuffle() is False


# TEST 5: Check that reshuffling is flagged at 75% penetration
def test_reshuffle_limit():
    shoe = Shoe(seed=42)

    shoe.deal(233)
    assert shoe.needs_reshuffle() is False

    # 234 out of 312 cards is exactly 75%
    shoe.deal(1)

    assert shoe.get_remaining() == 78
    assert shoe.get_penetration() == 0.75
    assert shoe.needs_reshuffle() is True


# TEST 6: Check that resetting restores a full shuffled shoe
def test_reset():
    shoe = Shoe(seed=42)

    shoe.deal(100)
    assert shoe.get_remaining() == 212

    shoe.reset()

    assert shoe.get_remaining() == 312
    assert shoe.get_penetration() == 0
    assert shoe.shuffle_count == 2


# TEST 7: Check that the same seed produces the same cards
def test_reproducibility():
    shoe1 = Shoe(seed=42)
    shoe2 = Shoe(seed=42)

    cards1 = [(card.value, card.suit) for card in shoe1.deal(20)]
    cards2 = [(card.value, card.suit) for card in shoe2.deal(20)]

    assert cards1 == cards2


# TEST 8: Check that six decks contain six copies of every card
def test_card_counts():
    shoe = Shoe(seed=42)

    cards = shoe.deal(312)
    counts = Counter((card.value, card.suit) for card in cards)

    # there are 52 unique rank-and-suit combinations
    assert len(counts) == 52

    # each combination appears six times
    assert all(count == 6 for count in counts.values())

    assert shoe.get_remaining() == 0


# TEST 9: Check that invalid deal requests are rejected
def test_invalid_deal():
    shoe = Shoe(seed=42)

    with pytest.raises(ValueError):
        shoe.deal(0)

    with pytest.raises(ValueError):
        shoe.deal(313)

    with pytest.raises(ValueError):
        shoe.deal(-1)

    with pytest.raises(ValueError):
        shoe.deal(1.5)

    # rejected deals should not remove any cards
    assert shoe.get_remaining() == 312


# TEST 10: Check that invalid shoe settings are rejected
def test_invalid_settings():
    with pytest.raises(ValueError):
        Shoe(num_decks=0)

    with pytest.raises(ValueError):
        Shoe(num_decks=-1)

    with pytest.raises(ValueError):
        Shoe(num_decks=1.5)

    with pytest.raises(ValueError):
        Shoe(penetration_limit=0)

    with pytest.raises(ValueError):
        Shoe(penetration_limit=1.5)


# TEST 11: Check that reaching penetration does not shuffle mid-round
def test_no_automatic_shuffle():
    shoe = Shoe(seed=42)

    shoe.deal(234)

    assert shoe.needs_reshuffle() is True
    assert shoe.shuffle_count == 1

    # the remaining cards can still be dealt
    shoe.deal(1)

    assert shoe.get_remaining() == 77
    assert shoe.shuffle_count == 1
