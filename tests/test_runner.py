from copy import deepcopy
from decimal import Decimal

import pydealer
import pytest

from blackjack.engine import BlackjackEngine
from blackjack.player import Player
from blackjack.shoe import Shoe
from strategies.basic import BasicStrategy
from strategies.flat import FlatBetting
from strategies.runner import run_round


# use a known card order for reliable game-engine tests
class ScenarioShoe(Shoe):
    def needs_reshuffle(self):
        return False


def scenario_engine(values, players=None):
    # PyDealer deals from the end of its stack
    shoe = ScenarioShoe(seed=1)
    shoe.shoe = pydealer.Stack(cards=[pydealer.Card(value, "Spades") for value in reversed(values)])
    return BlackjackEngine(players=players or {"p": Player(100)}, shoe=shoe)


# record the information Basic Strategy receives during the game
class RecordingBasic(BasicStrategy):
    def __init__(self):
        self.observations = []
        self.actions = []

    def choose_action(self, observation):
        self.observations.append(deepcopy(observation))
        action = super().choose_action(observation)
        self.actions.append(action)
        return action


# choose not to wager on any round
class NeverBet:
    def __init__(self):
        self.observations = []

    def choose_wager(self, observation):
        self.observations.append(deepcopy(observation))
        return None


# TEST 1: Check that flat betting and Basic Strategy complete a round
def test_complete_round():
    engine = scenario_engine(["10", "10", "8", "7"])
    playing = RecordingBasic()

    result = run_round(engine, {"p": playing}, {"p": FlatBetting(10)})

    assert result["status"] == "complete"
    assert result["wagers"] == {"p": Decimal("10")}
    assert playing.actions == ["stand"]
    assert engine.players["p"].bankroll == 110
    assert engine.players["p"].bets == {}


# TEST 2: Check that playing strategies never see the dealer's hidden card
def test_public_information_only():
    engine = scenario_engine(["10", "10", "8", "7"])
    playing = RecordingBasic()

    run_round(engine, {"p": playing}, {"p": FlatBetting(10)})

    assert len(playing.observations) == 1
    observation = playing.observations[0]
    assert observation["dealer"]["hole_hidden"] is True
    assert observation["dealer"]["total"] is None
    assert len(observation["dealer"]["cards"]) == 1
    assert "shoe" not in observation


# TEST 3: Check that natural blackjack requires no playing decisions
def test_natural_blackjack():
    engine = scenario_engine(["Ace", "10", "King", "7"])
    playing = RecordingBasic()

    result = run_round(engine, {"p": playing}, {"p": FlatBetting(10)})

    assert result["status"] == "complete"
    assert playing.actions == []
    assert engine.players["p"].bankroll == 115


# TEST 4: Check that two players can use separate strategies in the same game
def test_two_players():
    engine = scenario_engine(["10", "9", "6", "8", "7", "10", "2"],
                             players={"a": Player(100), "b": Player(100)})
    playing = {"a": RecordingBasic(), "b": RecordingBasic()}

    result = run_round(engine, playing, {"a": FlatBetting(10), "b": FlatBetting(10)})

    assert result["status"] == "complete"
    assert engine.round_number == 1
    assert len(engine.players["a"].hands) == 1
    assert len(engine.players["b"].hands) == 1
    assert all(player.bets == {} for player in engine.players.values())


# TEST 5: Check that sitting-out players are still given betting observations
def test_sitting_out():
    engine = scenario_engine(["10", "10", "8", "7"],
                             players={"observer": Player(100), "p": Player(100)})
    playing = {"observer": BasicStrategy(), "p": BasicStrategy()}
    observer = NeverBet()
    betting = {"observer": observer, "p": FlatBetting(10)}

    result = run_round(engine, playing, betting)

    assert result["wagers"] == {"p": Decimal("10")}
    assert engine.players["observer"].hands == []
    assert engine.players["observer"].bankroll == 100
    assert len(observer.observations) == 1
    assert observer.observations[0]["player_id"] == "observer"


# TEST 6: Check that the shoe stays unchanged if everyone sits out
def test_everyone_sits_out():
    engine = BlackjackEngine(players={"p": Player(100)}, seed=42)
    before_cards = list(engine.shoe.shoe.cards)
    before_random_state = engine.shoe.rng.getstate()

    result = run_round(engine, {"p": BasicStrategy()}, {"p": NeverBet()})

    assert result == {"status": "no_active_seats", "round_number": 0, "wagers": {}}
    assert engine.round_number == 0
    assert list(engine.shoe.shoe.cards) == before_cards
    assert engine.shoe.rng.getstate() == before_random_state


# TEST 7: Check that a compulsory card shortage returns a void round
def test_void_round():
    engine = scenario_engine(["10", "2", "8", "3"])

    result = run_round(engine, {"p": BasicStrategy()}, {"p": FlatBetting(10)})

    assert result["status"] == "void"
    assert engine.players["p"].bankroll == 100
    assert engine.players["p"].bets == {}


# TEST 8: Check that missing strategies are rejected before any cards are dealt
def test_missing_strategy():
    engine = BlackjackEngine(players={"a": Player(), "b": Player()}, seed=42)

    with pytest.raises(ValueError):
        run_round(engine, {"a": BasicStrategy()}, {"a": FlatBetting(10)})

    assert engine.round_number == 0
    assert engine.shoe.get_remaining() == 312


# TEST 9: Check that Flat Betting can play a series of full games automatically
def test_many_rounds():
    engine = BlackjackEngine(players={"p": Player(10000)}, seed=42)
    playing = {"p": BasicStrategy()}
    betting = {"p": FlatBetting(10)}

    for _ in range(100):
        result = run_round(engine, playing, betting)
        assert result["status"] == "complete"

    assert engine.round_number == 100
    assert len(engine.history.rounds) == 100
    assert engine.players["p"].bets == {}


# TEST 10: Check that splitting and doubling keep both players in the correct turn order
def test_split_double_two_players():
    # player a starts with 8 + 8 against dealer 10 and splits
    # player b has 10 + 9 and waits for player a to finish
    engine = scenario_engine(["8", "10", "10", "8", "9", "7", "3", "10", "9"],
                             players={"a": Player(100), "b": Player(100)})
    playing = {"a": RecordingBasic(), "b": RecordingBasic()}
    betting = {"a": FlatBetting(10), "b": FlatBetting(10)}

    result = run_round(engine, playing, betting)

    assert result["status"] == "complete"
    assert playing["a"].actions == ["split", "double", "stand"]
    assert playing["b"].actions == ["stand"]
    assert len(engine.players["a"].hands) == 2
    assert engine.players["a"].bankroll == 120
    assert engine.players["b"].bankroll == 110
