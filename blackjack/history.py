import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path


def card_record(card):
    # convert a PyDealer card into a simple dictionary
    return {"value": card.value, "suit": card.suit}

# record completed games for strategy analysis later
class History:
    def __init__(self, retain=True):
        # choose whether to save every completed round
        self.retain = retain

        # store completed round records
        self.rounds = []

        # always keep the most recent completed round
        self.last_round = None

    def append(self, record):
        # copy the record so later changes cannot affect it
        self.last_round = deepcopy(record)

        # save the record if history retention is enabled
        if self.retain:
            self.rounds.append(self.last_round)

    def export_json(self, path):
        # convert Decimal values into strings for JSON
        def encode(value):
            if isinstance(value, Decimal):
                return str(value)

            raise TypeError("Cannot export this value")

        # save the completed rounds to a JSON file
        Path(path).write_text(
            json.dumps(self.rounds, default=encode, indent=2) + "\n"
        )
