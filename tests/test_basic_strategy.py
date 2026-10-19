import json
from copy import deepcopy
from pathlib import Path

import pytest
from strategies.basic import BasicStrategy


# load an independently recorded six-deck S17 basic strategy chart
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "wizard_s17_das.json").read_text()
)
UPCARDS = REFERENCE["upcards"]


def observation_for(ranks, upcard="6", legal_actions=None):
    # create a public observation with the specified player and dealer cards
    values = {str(number): number for number in range(2, 11)}
    values.update({"Ace": 1, "Jack": 10, "Queen": 10, "King": 10})
    total = sum(values[rank] for rank in ranks)
    soft = "Ace" in ranks and total + 10 <= 21
    if soft:
        total += 10

    return {
        "state": "player_turn",
        "player_id": "p",
        "current_player": "p",
        "current_hand": "hand_1",
        "players": {"p": {"hands": [{
            "hand_id": "hand_1",
            "cards": [{"value": rank, "suit": "Spades"} for rank in ranks],
            "total": total,
            "soft": soft,
            "finished": False,
            "awaiting_card": False
        }]}},
        "dealer": {"cards": [{"value": upcard, "suit": "Hearts"}], "hole_hidden": True, "total": None},
        "legal_actions": ["hit", "stand", "double"] if legal_actions is None else legal_actions,
        "events": [{"type": "round_start", "num_decks": 6}]
    }


def expected_action(token, can_double=True):
    # translate the chart's recommendations into legal game actions
    if token == "H":
        return "hit"
    if token == "S":
        return "stand"
    if token == "P":
        return "split"
    if token == "Dh":
        return "double" if can_double else "hit"
    if token == "Ds":
        return "double" if can_double else "stand"
    raise ValueError("Unknown reference chart decision")


def hard_cards(total):
    # make two non-Ace cards adding up to the requested hard total
    first = min(10, total - 2)
    return [str(first), str(total - first)]


# TEST 1: Check every hard hand decision against the independent chart
@pytest.mark.parametrize("total,row", REFERENCE["hard"].items())
@pytest.mark.parametrize("column,upcard", list(enumerate(UPCARDS)))
@pytest.mark.parametrize("can_double", [True, False])
def test_hard_chart(total, row, column, upcard, can_double):
    legal = ["hit", "stand"] + (["double"] if can_double else [])
    observation = observation_for(hard_cards(int(total)), upcard, legal)
    token = row.split()[column]

    assert BasicStrategy().choose_action(observation) == expected_action(token, can_double)


# TEST 2: Check every soft hand decision and doubling fallback against the chart
@pytest.mark.parametrize("total,row", REFERENCE["soft"].items())
@pytest.mark.parametrize("column,upcard", list(enumerate(UPCARDS)))
@pytest.mark.parametrize("can_double", [True, False])
def test_soft_chart(total, row, column, upcard, can_double):
    ranks = ["Ace", str(int(total) - 11)]
    legal = ["hit", "stand"] + (["double"] if can_double else [])
    observation = observation_for(ranks, upcard, legal)
    token = row.split()[column]

    assert BasicStrategy().choose_action(observation) == expected_action(token, can_double)


# TEST 3: Check every pair decision against the independent chart
@pytest.mark.parametrize("rank,row", REFERENCE["pair"].items())
@pytest.mark.parametrize("column,upcard", list(enumerate(UPCARDS)))
@pytest.mark.parametrize("can_double", [True, False])
def test_pair_chart(rank, row, column, upcard, can_double):
    legal = ["hit", "stand", "split"] + (["double"] if can_double else [])
    observation = observation_for([rank, rank], upcard, legal)
    token = row.split()[column]

    assert BasicStrategy().choose_action(observation) == expected_action(token, can_double)


# TEST 4: Check that pairs use ordinary hand strategy when splitting is unavailable
@pytest.mark.parametrize("rank", REFERENCE["pair"])
@pytest.mark.parametrize("column,upcard", list(enumerate(UPCARDS)))
@pytest.mark.parametrize("can_double", [True, False])
def test_pairs_without_splitting(rank, column, upcard, can_double):
    observation = observation_for([rank, rank], upcard,
                                  ["hit", "stand"] + (["double"] if can_double else []))
    total = observation["players"]["p"]["hands"][0]["total"]

    # two Aces are a soft 12 if splitting is unavailable
    if rank == "Ace" or total < 9:
        token = "H"
    else:
        token = REFERENCE["hard"][str(total)].split()[column]

    assert BasicStrategy().choose_action(observation) == expected_action(token, can_double)


# TEST 5: Check that an Ace pair without splitting is treated as soft 12
@pytest.mark.parametrize("upcard", UPCARDS)
def test_ace_pair_without_split(upcard):
    observation = observation_for(["Ace", "Ace"], upcard, ["hit", "stand"])

    assert BasicStrategy().choose_action(observation) == "hit"


# TEST 6: Check that dealer face cards are treated as ten-value cards
@pytest.mark.parametrize("upcard", ["10", "Jack", "Queen", "King"])
def test_dealer_face_cards(upcard):
    observation = observation_for(["King", "Queen"], upcard, ["hit", "stand", "split"])
    assert BasicStrategy().choose_action(observation) == "stand"

    observation = observation_for(["5", "6"], upcard)
    assert BasicStrategy().choose_action(observation) == "double"


# TEST 7: Check that soft 18 stands when the recommended double is unavailable
@pytest.mark.parametrize("upcard", ["3", "4", "5", "6"])
def test_soft_18_fallback(upcard):
    observation = observation_for(["Ace", "7"], upcard, ["hit", "stand"])

    assert BasicStrategy().choose_action(observation) == "stand"


# TEST 8: Check that the active split hand does not need to be first in the list

def test_active_split_hand():
    observation = observation_for(["Ace", "2", "5"], "6", ["hit", "stand"])
    completed = deepcopy(observation["players"]["p"]["hands"][0])
    completed["hand_id"] = "previous_hand"
    completed["finished"] = True
    observation["players"]["p"]["hands"].insert(0, completed)

    assert BasicStrategy().choose_action(observation) == "stand"


# TEST 9: Check that the strategy does not modify the observation

def test_observation_not_changed():
    observation = observation_for(["8", "8"], "10", ["hit", "stand", "double", "split"])
    before = deepcopy(observation)

    assert BasicStrategy().choose_action(observation) == "split"
    assert observation == before


# TEST 10: Check that hidden or future information does not affect the decision

def test_no_private_information():
    observation = observation_for(["8", "8"], "10", ["hit", "stand", "double", "split"])
    other = deepcopy(observation)
    other["secret_hole_card"] = "Ace"
    other["future_cards"] = ["10", "10", "Ace"]

    assert BasicStrategy().choose_action(observation) == BasicStrategy().choose_action(other)


# TEST 11: Check that finished hands and invalid turns are rejected
@pytest.mark.parametrize("change", [
    lambda observation: observation.update(state="complete"),
    lambda observation: observation.update(current_player="other"),
    lambda observation: observation.update(current_hand="missing"),
    lambda observation: observation["players"]["p"]["hands"][0].update(finished=True),
    lambda observation: observation["players"]["p"]["hands"][0].update(awaiting_card=True),
    lambda observation: observation["players"]["p"]["hands"][0].update(total=19),
    lambda observation: observation["players"]["p"]["hands"][0].update(soft=True),
    lambda observation: observation.update(legal_actions=[]),
    lambda observation: observation.update(legal_actions=["surrender"]),
    lambda observation: observation.update(events=[]),
    lambda observation: observation["events"][0].update(num_decks=1),
    lambda observation: observation.update(dealer={"cards": []}),
    lambda observation: observation["dealer"]["cards"][0].update(value="Joker"),
])
def test_invalid_observations(change):
    observation = observation_for(["5", "6"])
    change(observation)

    with pytest.raises(ValueError):
        BasicStrategy().choose_action(observation)


# TEST 12: Check that completed or bust hands do not receive further actions
@pytest.mark.parametrize("ranks", [["Ace", "King"], ["7", "7", "7"], ["King", "Queen", "2"]])
def test_terminal_hands(ranks):
    with pytest.raises(ValueError):
        BasicStrategy().choose_action(observation_for(ranks))


# TEST 13: Check that an incomplete observation is rejected

def test_missing_fields():
    with pytest.raises(ValueError):
        BasicStrategy().choose_action({})


# TEST 14: Check that Basic Strategy can finish real seeded blackjack rounds

def test_complete_blackjack_rounds():
    # PyDealer is required only for this engine integration test
    pytest.importorskip("pydealer")
    from blackjack.engine import BlackjackEngine
    from blackjack.player import Player

    engine = BlackjackEngine(players={"p": Player(10000)}, seed=42)
    strategy = BasicStrategy()

    for _ in range(20):
        engine.start_round({"p": 10})

        while engine.state == "player_turn":
            observation = engine.observation("p")
            action = strategy.choose_action(observation)
            assert action in observation["legal_actions"]
            engine.act("p", action)

        assert engine.state == "complete"
        assert engine.players["p"].bets == {}

    assert len(engine.history.rounds) == 20
