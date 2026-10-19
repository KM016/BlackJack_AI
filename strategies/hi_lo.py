from copy import deepcopy
from decimal import Decimal

from strategies.betting import betting_limits, decimal_value


# assign a counting value to each card rank
CARD_VALUES = {str(number): 1 for number in range(2, 7)}
CARD_VALUES.update({str(number): 0 for number in range(7, 10)})
CARD_VALUES.update({rank: -1 for rank in ("10", "Jack", "Queen", "King", "Ace")})

# these public events reveal a card to the players
CARD_EVENTS = ("card", "dealer_upcard", "dealer_reveal", "dealer_card")


# track revealed cards to estimate whether the remaining shoe favours the player
class HiLoCounter:
    def __init__(self, num_decks=6):
        # check the number of decks and start from a neutral count
        if type(num_decks) is not int or num_decks < 1:
            raise ValueError("Number of decks must be a positive integer")

        self.num_decks = num_decks
        self.running_count = 0
        self.cards_seen = 0

        # remember previously counted events to avoid counting cards twice
        self.last_round = 0
        self.last_events = []
        self.round_finished = True

    def get_decks_remaining(self):
        # estimate remaining decks from cards that have been revealed
        remaining_cards = 52 * self.num_decks - self.cards_seen
        return Decimal(remaining_cards) / Decimal(52)

    def get_true_count(self):
        # account for how many cards are estimated to remain
        remaining_cards = 52 * self.num_decks - self.cards_seen
        if remaining_cards <= 0:
            raise ValueError("No cards remaining for true count")

        return Decimal(self.running_count * 52) / Decimal(remaining_cards)

    def observe(self, observation):
        # read the public events for the latest round
        try:
            round_number = observation["round_number"]
            events = observation["events"]
            num_decks = observation.get("num_decks", self.num_decks)

            if type(round_number) is not int or round_number < 0 or not isinstance(events, list):
                raise ValueError("Invalid round information")
            if type(num_decks) is not int or num_decks != self.num_decks:
                raise ValueError("Wrong number of decks")

            # an empty game has no cards to count
            if round_number == 0:
                if events or self.last_round != 0:
                    raise ValueError("Invalid initial round")
                return

            # we must see rounds in order so that we never miss revealed cards
            if round_number not in (self.last_round, self.last_round + 1):
                raise ValueError("Missed or out-of-order round")

            same_round = round_number == self.last_round
            if not same_round and not self.round_finished:
                raise ValueError("Previous round was not fully observed")
            if not events or events[0]["type"] != "round_start":
                raise ValueError("Missing round start")

            # check that the public events belong to the correct game
            start = events[0]
            if (type(start["round_number"]) is not int or start["round_number"] != round_number
                    or type(start["num_decks"]) is not int or start["num_decks"] != self.num_decks
                    or type(start["reshuffled"]) is not bool):
                raise ValueError("Invalid round start")

            # repeated observations must contain the same events already seen
            previous = self.last_events if same_round else []
            if events[:len(previous)] != previous:
                raise ValueError("Previously observed events have changed")
            if same_round and self.round_finished and events != previous:
                raise ValueError("Completed round cannot gain new events")

            # calculate changes first, without altering the stored counter
            count = self.running_count
            cards_seen = self.cards_seen
            finished = self.round_finished if same_round else False

            for index in range(len(previous), len(events)):
                event = events[index]
                kind = event["type"]

                if finished:
                    raise ValueError("Events cannot follow the end of a round")

                if kind == "round_start":
                    if index != 0:
                        raise ValueError("Unexpected round start")
                    if event["reshuffled"]:
                        # a fresh shoe starts with no previously counted cards
                        count = 0
                        cards_seen = 0

                elif kind in CARD_EVENTS:
                    # only count cards actually revealed to the players
                    rank = event["card"]["value"]
                    if rank not in CARD_VALUES:
                        raise ValueError("Unknown card rank")
                    count += CARD_VALUES[rank]
                    cards_seen += 1

                elif kind == "round_end":
                    finished = True

            if cards_seen > self.num_decks * 52:
                raise ValueError("More visible cards than the shoe contains")
            if observation.get("state") in ("complete", "void") and not finished:
                raise ValueError("Completed round is missing its ending")

            # save the validated count and event history
            self.running_count = count
            self.cards_seen = cards_seen
            self.last_round = round_number
            self.last_events = deepcopy(events)
            self.round_finished = finished

        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise ValueError("Invalid public card events") from error

    def get_betting_true_count(self, observation):
        # count the cards revealed in the previous round
        self.observe(observation)

        if self.last_round > 0 and not self.round_finished:
            raise ValueError("Cannot bet before the previous round ends")

        # if the next round will reshuffle, the old count cannot guide the bet
        pending = observation["reshuffle_pending"]
        if type(pending) is not bool:
            raise ValueError("Invalid reshuffle signal")
        if pending:
            return Decimal(0)

        return self.get_true_count()


# adjust the wager based on the Hi-Lo true count
class HiLoBetting:
    def __init__(self, base_unit=None, counter=None):
        # default to the table minimum if no base stake is provided
        self.base_unit = None if base_unit is None else decimal_value(base_unit, "base unit")
        if self.base_unit is not None and self.base_unit <= 0:
            raise ValueError("Base unit must be positive")

        self.counter = counter if counter is not None else HiLoCounter()
        self.last_decision = None

    def get_multiplier(self, true_count):
        # use a fixed betting spread from 1 to 8 units
        if true_count >= 4:
            return 8
        if true_count >= 3:
            return 6
        if true_count >= 2:
            return 4
        if true_count >= 1:
            return 2
        return 1

    def choose_wager(self, observation):
        # read the available bankroll and casino betting limits
        bankroll, minimum, maximum, can_bet = betting_limits(observation)

        # update the card count before choosing the next wager
        true_count = self.counter.get_betting_true_count(observation)
        base_unit = minimum if self.base_unit is None else self.base_unit
        if base_unit < minimum:
            raise ValueError("Base unit is below the minimum bet")

        # increase the stake when the true count is favourable
        proposed_wager = base_unit * self.get_multiplier(true_count)

        # never bet more than the table limit or available bankroll
        wager = min(proposed_wager, maximum, bankroll)
        if not can_bet or bankroll < minimum or wager < minimum:
            wager = None

        # store the decision so we can analyse the strategy later
        self.last_decision = {
            "true_count": true_count,
            "base_unit": base_unit,
            "proposed_wager": proposed_wager,
            "wager": wager,
            "capped": wager is not None and wager < proposed_wager,
            "sit_out_reason": "insufficient_bankroll" if wager is None else None
        }

        return wager
