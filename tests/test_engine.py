from copy import deepcopy
from decimal import Decimal
import json

import pydealer
import pytest

from blackjack.engine import BlackjackEngine, ShoeExhausted
from blackjack.player import Player
from blackjack.rules import Rules
from blackjack.shoe import Shoe


# create a shoe with a fixed card order for testing complete rounds
class ScenarioShoe(Shoe):
    def needs_reshuffle(self):
        return False


def make_engine(*values, players=None, bankroll=100, rules=None):
    # PyDealer deals from the end of the stack, so reverse our card order
    rules = rules if rules is not None else Rules()
    shoe = ScenarioShoe(rules.num_decks, seed=1, penetration_limit=rules.penetration)
    cards = [pydealer.Card(value, "Spades") for value in reversed(values)]
    shoe.shoe = pydealer.Stack(cards=cards)

    return BlackjackEngine(
        players=players if players is not None else {"p": Player(bankroll)},
        rules=rules,
        shoe=shoe
    )


def finish_round(engine):
    # stand on every remaining player hand
    while engine.state == "player_turn":
        engine.act(engine.current_player, "stand")

    assert engine.state == "complete"


# TEST 1: Check that the engine starts ready for betting

def test_initial_engine():
    engine = BlackjackEngine(seed=42)

    assert engine.state == "betting"
    assert engine.round_number == 0
    assert engine.current_player is None
    assert engine.shoe.get_remaining() == 312
    assert list(engine.players) == ["player_1"]


# TEST 2: Check that the cards are dealt in table seat order

def test_dealing_order():
    players = {"a": Player(100), "b": Player(100)}
    engine = make_engine("2", "3", "6", "4", "5", "10", "2", players=players)

    # the order of players at the table takes priority over wager dictionary order
    engine.start_round({"b": 10, "a": 10})

    assert [c.value for c in players["a"].hands[0].hand.cards] == ["2", "4"]
    assert [c.value for c in players["b"].hands[0].hand.cards] == ["3", "5"]
    assert [c.value for c in engine.dealer.hand.cards] == ["6", "10"]
    assert engine.current_player == "a"

    engine.act("a", "stand")
    assert engine.current_player == "b"
    engine.act("b", "stand")
    assert engine.state == "complete"


# TEST 3: Check that normal wins, losses and pushes pay correctly
@pytest.mark.parametrize("cards,outcome,bankroll", [
    (("10", "10", "8", "7"), "win", 110),
    (("10", "10", "7", "8"), "loss", 90),
    (("10", "10", "8", "8"), "push", 100),
])
def test_normal_payouts(cards, outcome, bankroll):
    engine = make_engine(*cards)
    engine.start_round({"p": 10})
    engine.act("p", "stand")

    hand = engine.players["p"].hands[0]
    assert hand.outcome == outcome
    assert engine.players["p"].bankroll == bankroll
    assert engine.players["p"].bets == {}
    assert engine.state == "complete"


# TEST 4: Check that doubling down increases both the wager and payout

def test_double_down():
    engine = make_engine("5", "10", "6", "8", "10")
    engine.start_round({"p": 10})

    assert "double" in engine.legal_actions("p")
    engine.act("p", "double")

    hand = engine.players["p"].hands[0]
    assert hand.doubled is True
    assert hand.outcome == "win"
    assert hand.net_profit == 20
    assert engine.players["p"].bankroll == 120
    assert engine.wagers[hand.hand_id] == 20


# TEST 5: Check that a dealer natural blackjack ends the round immediately

def test_dealer_blackjack():
    engine = make_engine("10", "Ace", "9", "King")
    engine.start_round({"p": 10})

    assert engine.state == "complete"
    assert engine.dealer.revealed is True
    assert engine.players["p"].hands[0].outcome == "loss"
    assert engine.players["p"].bankroll == 90


# TEST 6: Check that simultaneous natural blackjacks push

def test_both_blackjack():
    engine = make_engine("Ace", "Ace", "King", "King")
    engine.start_round({"p": 10})

    assert engine.state == "complete"
    assert engine.players["p"].hands[0].outcome == "push"
    assert engine.players["p"].bankroll == 100


# TEST 7: Check that a player natural blackjack pays 3:2

def test_player_blackjack():
    engine = make_engine("Ace", "10", "King", "7")
    engine.start_round({"p": 10})

    assert engine.state == "complete"
    assert engine.players["p"].hands[0].outcome == "blackjack"
    assert engine.players["p"].bankroll == 115


# TEST 8: Check that a failed dealer peek keeps the hole card hidden

def test_hidden_dealer_card():
    engine = make_engine("8", "Ace", "7", "6")
    engine.start_round({"p": 10})
    observation = engine.observation("p")

    assert observation["dealer"]["cards"] == [{"value": "Ace", "suit": "Spades"}]
    assert observation["dealer"]["hole_hidden"] is True
    assert observation["dealer"]["total"] is None
    assert len(engine.dealer.hand.cards) == 2

    # changing the observation should not change the game
    observation["dealer"]["cards"].clear()
    observation["events"].clear()
    assert engine.dealer.get_upcard().value == "Ace"
    assert engine.public_events

    finish_round(engine)
    assert engine.observation("p")["dealer"]["hole_hidden"] is False


# TEST 9: Check that hidden and future cards cannot change the public observation

def test_no_future_card_leak():
    first = make_engine("8", "10", "7", "6", "2")
    second = make_engine("8", "10", "7", "9", "King")

    first.start_round({"p": 10})
    second.start_round({"p": 10})

    assert first.observation("p") == second.observation("p")


# TEST 10: Check that dealer stands on soft 17

def test_dealer_soft_17():
    engine = make_engine("10", "Ace", "8", "6")
    engine.start_round({"p": 10})
    finish_round(engine)

    assert engine.dealer.hand.get_total() == 17
    assert engine.dealer.hand.is_soft() is True


# TEST 11: Check that hitting to bust finishes and settles a hand

def test_player_bust():
    engine = make_engine("10", "6", "9", "7", "King")
    engine.start_round({"p": 10})
    engine.act("p", "hit")

    assert engine.players["p"].hands[0].hand.is_bust() is True
    assert engine.players["p"].hands[0].outcome == "loss"
    assert engine.players["p"].bankroll == 90
    assert engine.state == "complete"

    # the dealer did not draw because all hands were already settled
    assert len(engine.dealer.hand.cards) == 2


# TEST 12: Check that a split creates two hands with independent wagers

def test_split_and_double():
    engine = make_engine("8", "10", "8", "7", "3", "9", "10")
    engine.start_round({"p": 10})
    engine.act("p", "split")

    assert len(engine.players["p"].hands) == 2
    assert engine.players["p"].bankroll == 80
    assert engine.current_hand.hand.get_total() == 11

    # double the first hand before the second receives its replacement card
    engine.act("p", "double")
    assert engine.current_hand.hand.get_total() == 18
    engine.act("p", "stand")

    assert [h.outcome for h in engine.players["p"].hands] == ["win", "win"]
    assert engine.players["p"].bankroll == 130
    assert [h["wager"] for h in engine.history.last_round["players"]["p"]["hands"]] == [20, 10]


# TEST 13: Check that the second split hand waits until its turn

def test_split_replacement_order():
    engine = make_engine("8", "10", "8", "7", "3", "4", "9")
    engine.start_round({"p": 10})
    engine.act("p", "split")
    first, second = engine.players["p"].hands

    assert [c.value for c in first.hand.cards] == ["8", "3"]
    assert [c.value for c in second.hand.cards] == ["8"]
    assert second.awaiting_card is True

    # playing the first hand happens before drawing the second replacement
    engine.act("p", "stand")
    assert [c.value for c in second.hand.cards] == ["8", "4"]
    assert second.awaiting_card is False
    finish_round(engine)


# TEST 14: Check that resplitting can create four hands but no more

def test_resplit_limit():
    engine = make_engine("8", "10", "8", "7", "8", "8", "8", "8", "8", "8")
    engine.start_round({"p": 10})

    for _ in range(3):
        engine.act("p", "split")

    assert len(engine.players["p"].hands) == 4
    assert engine.players["p"].bankroll == 60
    assert "split" not in engine.legal_actions("p")
    finish_round(engine)


# TEST 15: Check that split Aces receive one card and immediately finish

def test_split_aces():
    engine = make_engine("Ace", "10", "Ace", "7", "King", "9")
    engine.start_round({"p": 10})
    engine.act("p", "split")

    assert engine.state == "complete"
    assert len(engine.players["p"].hands) == 2
    assert all(h.finished and h.split_aces for h in engine.players["p"].hands)
    assert all(not h.hand.is_blackjack() for h in engine.players["p"].hands)
    assert engine.players["p"].bankroll == 120


# TEST 16: Check that 21 after splitting is not a natural blackjack

def test_split_twenty_one():
    engine = make_engine("King", "10", "King", "8", "Ace", "Ace")
    engine.start_round({"p": 10})
    engine.act("p", "split")

    assert engine.state == "complete"
    assert engine.players["p"].bankroll == 120
    assert all(h.outcome == "win" for h in engine.players["p"].hands)
    assert all(not h.hand.is_blackjack() for h in engine.players["p"].hands)


# TEST 17: Check that illegal actions cannot change the current round

def test_illegal_action():
    engine = make_engine("5", "10", "6", "8")
    engine.start_round({"p": 10})
    before = deepcopy((engine.observation("p"), engine.players["p"].bets,
                       list(engine.shoe.shoe.cards)))

    for action in ("split", "surrender", "insurance"):
        with pytest.raises(ValueError):
            engine.act("p", action)

    with pytest.raises(ValueError):
        engine.act("unknown", "stand")

    after = deepcopy((engine.observation("p"), engine.players["p"].bets,
                      list(engine.shoe.shoe.cards)))
    assert before == after
    finish_round(engine)


# TEST 18: Check that invalid bets do not remove money or cards
@pytest.mark.parametrize("amount", [0, -1, "NaN", "bad", "0.99", 1001, 101])
def test_invalid_wagers(amount):
    engine = make_engine("8", "10", "7", "8")

    with pytest.raises(ValueError):
        engine.start_round({"p": amount})

    assert engine.players["p"].bankroll == 100
    assert engine.shoe.get_remaining() == 4
    assert engine.round_number == 0


# TEST 19: Check that bets from multiple players are validated together

def test_wager_validation_before_deduction():
    players = {"a": Player(100), "b": Player(5)}
    engine = make_engine("2", "3", "10", "4", "5", "7", players=players)

    with pytest.raises(ValueError):
        engine.start_round({"a": 10, "b": 10})

    assert players["a"].bankroll == 100
    assert players["b"].bankroll == 5
    assert engine.shoe.get_remaining() == 6


# TEST 20: Check that the engine rejects an initial deal without enough cards

def test_initial_shoe_exhaustion():
    engine = make_engine("8", "10", "7")

    with pytest.raises(ShoeExhausted):
        engine.start_round({"p": 10})

    assert engine.round_number == 0
    assert engine.players["p"].bankroll == 100
    assert engine.shoe.get_remaining() == 3


# TEST 21: Check that an unavailable optional hit is rejected without changes

def test_optional_card_exhaustion():
    engine = make_engine("8", "10", "7", "8")
    engine.start_round({"p": 10})

    before = deepcopy((engine.observation("p"), engine.players["p"].bankroll,
                       engine.public_events))
    with pytest.raises(ShoeExhausted):
        engine.act("p", "hit")
    after = deepcopy((engine.observation("p"), engine.players["p"].bankroll,
                      engine.public_events))

    assert before == after


# TEST 22: Check that a failed compulsory split draw refunds every wager

def test_required_split_card_exhaustion():
    engine = make_engine("8", "10", "8", "7", "3")
    engine.start_round({"p": 10})
    engine.act("p", "split")

    # the second hand now needs a replacement, but the shoe is empty
    with pytest.raises(ShoeExhausted):
        engine.act("p", "stand")

    assert engine.state == "void"
    assert engine.players["p"].bankroll == 100
    assert engine.players["p"].bets == {}
    assert all(h.outcome == "void" for h in engine.players["p"].hands)
    assert engine.history.last_round["state"] == "void"


# TEST 23: Check that running out of dealer cards refunds the round

def test_required_dealer_card_exhaustion():
    engine = make_engine("10", "2", "8", "3")
    engine.start_round({"p": 10})

    with pytest.raises(ShoeExhausted):
        engine.act("p", "stand")

    assert engine.state == "void"
    assert engine.players["p"].bankroll == 100
    assert engine.players["p"].bets == {}


# TEST 24: Check that the shoe is reshuffled only between rounds

def test_between_round_shuffle():
    engine = BlackjackEngine({"p": Player(1000)}, rules=Rules(penetration=0.01), seed=22)
    engine.start_round({"p": 10})
    shuffle_count = engine.shoe.shuffle_count
    finish_round(engine)

    assert engine.shoe.shuffle_count == shuffle_count

    engine.start_round({"p": 10})
    assert engine.shoe.shuffle_count == shuffle_count + 1
    finish_round(engine)


# TEST 25: Check that only participating players are dealt cards

def test_sitting_out():
    players = {"a": Player(100), "b": Player(100)}
    engine = make_engine("10", "10", "8", "7", players=players)
    engine.start_round({"a": 10})
    finish_round(engine)

    assert players["b"].hands == []
    assert players["b"].bankroll == 100
    assert engine.history.last_round["players"]["b"]["initial_wager"] == 0


# TEST 26: Check that betting observations do not deal any cards

def test_betting_observation():
    engine = make_engine("8", "10", "7", "8")
    before = engine.shoe.get_remaining()
    observation = engine.betting_observation("p")

    assert observation["can_bet"] is True
    assert observation["bankroll"] == 100
    assert observation["next_round_number"] == 1
    assert engine.shoe.get_remaining() == before

    engine.start_round({"p": 10})
    with pytest.raises(ValueError):
        engine.betting_observation("p")


# TEST 27: Check that completed rounds are recorded and can be exported

def test_history_record(tmp_path):
    engine = make_engine("10", "10", "8", "7")
    engine.start_round({"p": 10})
    finish_round(engine)

    record = engine.history.last_round
    assert record["state"] == "complete"
    assert record["players"]["p"]["net_profit"] == 10
    assert record["players"]["p"]["hands"][0]["outcome"] == "win"
    assert len(record["dealer_cards"]) == 2

    path = tmp_path / "rounds.json"
    engine.history.export_json(path)
    exported = json.loads(path.read_text())
    assert exported[0]["players"]["p"]["net_profit"] == "10"


# TEST 28: Check that round history can be disabled during large simulations

def test_no_history_retention():
    engine = BlackjackEngine(seed=1, retain_history=False)

    engine.start_round({"player_1": 10})
    finish_round(engine)

    assert engine.history.rounds == []
    assert engine.history.last_round is not None


# TEST 29: Check that the same seed produces the same round results

def test_reproducible_rounds():
    first = BlackjackEngine(seed=42)
    second = BlackjackEngine(seed=42)

    for _ in range(10):
        for engine in (first, second):
            engine.start_round({"player_1": 10})
            finish_round(engine)

    assert first.players["player_1"].bankroll == second.players["player_1"].bankroll
    assert first.history.rounds == second.history.rounds


# TEST 30: Check that the table rejects invalid player configurations

def test_invalid_table():
    with pytest.raises(ValueError):
        BlackjackEngine({})

    with pytest.raises(ValueError):
        BlackjackEngine({str(i): Player() for i in range(8)})

    same_player = Player()
    with pytest.raises(ValueError):
        BlackjackEngine({"a": same_player, "b": same_player})

    with pytest.raises(ValueError):
        BlackjackEngine(shoe=Shoe(1))
