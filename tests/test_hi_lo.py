from copy import deepcopy
from decimal import Decimal

import pydealer
import pytest

from blackjack.engine import BlackjackEngine
from blackjack.player import Player
from blackjack.shoe import Shoe
from strategies.basic import BasicStrategy
from strategies.flat import FlatBetting
from strategies.hi_lo import HiLoBetting, HiLoCounter
from strategies.runner import run_round


def betting_view(**changes):
    # create a public observation for testing pre-round wagers
    view = {
        "state": "betting",
        "round_number": 0,
        "bankroll": Decimal("1000"),
        "min_bet": Decimal("1"),
        "max_bet": Decimal("1000"),
        "can_bet": True,
        "num_decks": 6,
        "reshuffle_pending": False,
        "events": []
    }
    view.update(changes)
    return view


def public_events(ranks=(), round_number=1, reshuffled=False, complete=True, num_decks=6):
    # create an ordered record of cards revealed during a round
    events = [{
        "type": "round_start", "round_number": round_number,
        "num_decks": num_decks, "reshuffled": reshuffled
    }]
    for rank in ranks:
        events.append({"type": "card", "card": {"value": rank, "suit": "Spades"}})
    if complete:
        events.append({"type": "round_end", "state": "complete"})
    return events


# TEST 1: Check that low, middle and high cards change the count correctly
@pytest.mark.parametrize("rank,expected", [
    ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1),
    ("7", 0), ("8", 0), ("9", 0),
    ("10", -1), ("Jack", -1), ("Queen", -1), ("King", -1), ("Ace", -1)
])
def test_card_values(rank, expected):
    counter = HiLoCounter()
    counter.observe(betting_view(round_number=1, state="complete",
                                 events=public_events([rank])))

    assert counter.running_count == expected
    assert counter.cards_seen == 1


# TEST 2: Check that the true count adjusts for remaining decks
def test_true_count():
    counter = HiLoCounter()
    ranks = ["2", "4", "5", "King", "Ace", "9"]

    counter.observe(betting_view(round_number=1, state="complete",
                                 events=public_events(ranks)))

    assert counter.running_count == 1
    assert counter.cards_seen == 6
    assert counter.get_decks_remaining() == Decimal(306) / 52
    assert counter.get_true_count() == Decimal(52) / 306


# TEST 3: Check that a complete deck has a balanced count
def test_balanced_deck():
    ranks = ([str(number) for number in range(2, 11)]
             + ["Jack", "Queen", "King", "Ace"]) * 4
    counter = HiLoCounter()
    counter.observe(betting_view(round_number=1, state="complete",
                                 events=public_events(ranks)))

    assert counter.running_count == 0
    assert counter.cards_seen == 52
    assert counter.get_decks_remaining() == 5


# TEST 4: Check that observing the same cards twice does not count them twice
def test_no_double_counting():
    counter = HiLoCounter()
    observation = betting_view(round_number=1, state="complete",
                               events=public_events(["2", "3", "King"]))

    counter.observe(observation)
    counter.observe(observation)

    assert counter.running_count == 1
    assert counter.cards_seen == 3


# TEST 5: Check that new public events are counted during a round
def test_new_events():
    counter = HiLoCounter()
    events = public_events(["2", "3"], complete=False)

    counter.observe(betting_view(round_number=1, state="player_turn", events=events))
    assert counter.running_count == 2
    assert counter.cards_seen == 2

    events += [{"type": "dealer_reveal", "card": {"value": "Ace"}},
               {"type": "round_end", "state": "complete"}]
    counter.observe(betting_view(round_number=1, state="complete", events=events))

    assert counter.running_count == 1
    assert counter.cards_seen == 3


# TEST 6: Check that actions, bets and settlements do not count as cards
def test_non_card_events():
    counter = HiLoCounter()
    events = public_events(["2"], complete=False)
    events += [{"type": "bet", "amount": 10}, {"type": "action", "action": "hit"},
               {"type": "settlement", "outcome": "win"},
               {"type": "round_end", "state": "complete"}]
    counter.observe(betting_view(round_number=1, state="complete", events=events))

    assert counter.running_count == 1
    assert counter.cards_seen == 1


# TEST 7: Check that the dealer's hidden card is counted only when revealed
class ScenarioShoe(Shoe):
    def needs_reshuffle(self):
        return False


def test_hidden_dealer_card():
    shoe = ScenarioShoe(seed=1)
    ranks = ["10", "6", "8", "Ace"]
    shoe.shoe = pydealer.Stack(cards=[pydealer.Card(rank, "Spades") for rank in reversed(ranks)])
    engine = BlackjackEngine({"p": Player(100)}, shoe=shoe)
    engine.start_round({"p": 10})

    counter = HiLoCounter()
    counter.observe(engine.observation("p"))

    assert counter.cards_seen == 3
    assert counter.running_count == 0
    assert engine.dealer.revealed is False

    engine.act("p", "stand")
    counter.observe(engine.observation("p"))
    counter.observe(engine.observation("p"))

    assert counter.cards_seen == 4
    assert counter.running_count == -1


# TEST 8: Check that the counter resets when the shoe is reshuffled
def test_shuffle_reset():
    counter = HiLoCounter()
    counter.observe(betting_view(round_number=1, state="complete",
                                 events=public_events(["2", "3"])))
    counter.observe(betting_view(round_number=2, state="complete",
                                 events=public_events(["King"], 2, reshuffled=True)))

    assert counter.running_count == -1
    assert counter.cards_seen == 1


# TEST 9: Check that a pending shuffle uses a neutral count for the next bet
def test_pending_shuffle():
    policy = HiLoBetting()
    view = betting_view(round_number=1, state="complete", reshuffle_pending=True,
                        events=public_events(["2"] * 20))

    assert policy.choose_wager(view) == 1
    assert policy.last_decision["true_count"] == 0
    assert policy.counter.running_count == 20


# TEST 10: Check that the betting spread follows the agreed true-count thresholds
@pytest.mark.parametrize("count,units", [
    ("-3", 1), ("0", 1), ("0.99", 1), ("1", 2), ("1.99", 2),
    ("2", 4), ("2.99", 4), ("3", 6), ("3.99", 6), ("4", 8), ("8", 8)
])
def test_bet_spread(count, units):
    assert HiLoBetting().get_multiplier(Decimal(count)) == units


# TEST 11: Check that a favourable count increases the wager
def test_higher_bet():
    policy = HiLoBetting()
    # 24 low cards give a true count of more than 4
    view = betting_view(round_number=1, state="complete",
                        events=public_events(["2", "3"] * 12))

    assert policy.choose_wager(view) == 8
    assert policy.last_decision["proposed_wager"] == 8


# TEST 12: Check that table and bankroll limits cap the wager
@pytest.mark.parametrize("bankroll,maximum,expected", [
    ("1000", "5", "5"),
    ("3.25", "1000", "3.25"),
    ("0.99", "1000", None),
])
def test_bet_caps(bankroll, maximum, expected):
    policy = HiLoBetting()
    view = betting_view(round_number=1, state="complete",
                        events=public_events(["2", "3"] * 12),
                        bankroll=Decimal(bankroll), max_bet=Decimal(maximum),
                        can_bet=Decimal(bankroll) >= 1)

    wager = policy.choose_wager(view)
    assert wager == (Decimal(expected) if expected is not None else None)
    assert policy.last_decision["capped"] == (expected is not None and expected != "8")


# TEST 13: Check that we can choose a different base betting unit
def test_custom_base_unit():
    policy = HiLoBetting("2.50")

    assert policy.choose_wager(betting_view()) == Decimal("2.50")


# TEST 14: Check that invalid base betting units are rejected
@pytest.mark.parametrize("amount", [0, -1, "NaN", "Infinity", True, 1.5])
def test_invalid_base_unit(amount):
    with pytest.raises(ValueError):
        HiLoBetting(amount)


# TEST 15: Check that a base unit below the table minimum is rejected
def test_base_below_minimum():
    with pytest.raises(ValueError):
        HiLoBetting("0.50").choose_wager(betting_view())


# TEST 16: Check that skipping a round is not allowed
def test_skipped_round():
    counter = HiLoCounter()
    counter.observe(betting_view(round_number=1, state="complete",
                                 events=public_events(["2"])))
    with pytest.raises(ValueError):
        counter.observe(betting_view(round_number=3, state="complete",
                                     events=public_events(["King"], 3)))


# TEST 17: Check that an unfinished round cannot be skipped
def test_incomplete_round():
    counter = HiLoCounter()
    counter.observe(betting_view(round_number=1, state="player_turn",
                                 events=public_events(["2"], complete=False)))
    with pytest.raises(ValueError):
        counter.observe(betting_view(round_number=2, state="complete",
                                     events=public_events(["3"], 2)))


# TEST 18: Check that changed card history is rejected without changing the count
def test_changed_history():
    counter = HiLoCounter()
    original = betting_view(round_number=1, state="complete",
                            events=public_events(["2"]))
    counter.observe(original)

    with pytest.raises(ValueError):
        counter.observe(betting_view(round_number=1, state="complete",
                                     events=public_events(["King"])))
    assert counter.running_count == 1
    assert counter.cards_seen == 1


# TEST 19: Check that invalid card events leave the count unchanged
def test_invalid_card():
    counter = HiLoCounter()
    events = public_events(["2", "not_a_card"])

    with pytest.raises(ValueError):
        counter.observe(betting_view(round_number=1, state="complete", events=events))

    assert counter.running_count == 0
    assert counter.cards_seen == 0


# TEST 20: Check that an empty shoe cannot produce a true count
def test_empty_shoe_true_count():
    counter = HiLoCounter(num_decks=1)
    ranks = ([str(number) for number in range(2, 11)]
             + ["Jack", "Queen", "King", "Ace"]) * 4
    counter.observe(betting_view(num_decks=1, round_number=1, state="complete",
                                 events=public_events(ranks, num_decks=1)))

    with pytest.raises(ValueError):
        counter.get_true_count()

    # a publicly announced reshuffle means we use a neutral count instead
    view = betting_view(num_decks=1, round_number=1, state="complete",
                        reshuffle_pending=True, events=public_events(ranks, num_decks=1))
    assert counter.get_betting_true_count(view) == 0


# TEST 21: Check that Hi-Lo plays complete rounds through the real runner
def test_real_game_integration():
    engine = BlackjackEngine({"p": Player(10000)}, seed=42)
    policy = HiLoBetting()
    playing = {"p": BasicStrategy()}
    betting = {"p": policy}

    for _ in range(25):
        result = run_round(engine, playing, betting)
        assert result["status"] == "complete"

    # the final round's exposed cards are collected at the next betting decision
    policy.choose_wager(engine.betting_observation("p"))
    assert policy.counter.last_round == 25
    assert engine.players["p"].bets == {}
    assert policy.counter.cards_seen > 0


# TEST 22: Check that observing an event list never changes the original data
def test_observation_unchanged():
    counter = HiLoCounter()
    view = betting_view(round_number=1, state="complete", events=public_events(["2", "Ace"]))
    before = deepcopy(view)

    counter.observe(view)

    assert view == before
