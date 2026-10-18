
import json
from decimal import Decimal

import pydealer

from blackjack.history import History, card_record


# TEST 1: Check that a PyDealer card is converted into a dictionary
def test_card_record():
    card = pydealer.Card("Ace", "Spades")

    assert card_record(card) == {
        "value": "Ace",
        "suit": "Spades"
    }


# TEST 2: Check that completed rounds are stored
def test_history():
    history = History()

    history.append({"round": 1, "profit": Decimal("10")})
    history.append({"round": 2, "profit": Decimal("-5")})

    assert len(history.rounds) == 2
    assert history.last_round["round"] == 2


# TEST 3: Check that changing the original record does not change history
def test_history_copy():
    history = History()
    record = {"cards": ["Ace", "King"]}

    history.append(record)

    # change the original record after storing it
    record["cards"].append("5")

    assert history.last_round["cards"] == ["Ace", "King"]


# TEST 4: Check that history can be disabled for large simulations
def test_no_retention():
    history = History(retain=False)

    history.append({"round": 1})

    assert history.rounds == []
    assert history.last_round == {"round": 1}


# TEST 5: Check that records can be exported to JSON
def test_json_export(tmp_path):
    history = History()

    history.append({
        "round": 1,
        "profit": Decimal("10.50")
    })

    path = tmp_path / "history.json"
    history.export_json(path)

    # check that the JSON file contains the correct data
    data = json.loads(path.read_text())

    assert data[0]["round"] == 1
    assert data[0]["profit"] == "10.50"
