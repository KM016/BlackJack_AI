# six-deck basic strategy: dealer stands on soft 17 and doubling after splitting is allowed
# chart reference: https://wizardofodds.com/games/blackjack/strategy/calculator/

# dealer upcards are ordered from 2 to Ace
UPCARDS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "Ace")

# H = hit, S = stand, D = double (otherwise hit), B = double (otherwise stand)
HARD = {
    9:  "HDDDDHHHHH",
    10: "DDDDDDDDHH",
    11: "DDDDDDDDDH",
    12: "HHSSSHHHHH",
    13: "SSSSSHHHHH",
    14: "SSSSSHHHHH",
    15: "SSSSSHHHHH",
    16: "SSSSSHHHHH",
}

SOFT = {
    12: "HHHHHHHHHH",
    13: "HHHDDHHHHH",
    14: "HHHDDHHHHH",
    15: "HHDDDHHHHH",
    16: "HHDDDHHHHH",
    17: "HDDDDHHHHH",
    18: "SBBBBSSHHH",
}

# these pairs should be split against the listed dealer upcards
SPLIT = {
    "Ace": UPCARDS,
    "2": ("2", "3", "4", "5", "6", "7"),
    "3": ("2", "3", "4", "5", "6", "7"),
    "4": ("5", "6"),
    "6": ("2", "3", "4", "5", "6"),
    "7": ("2", "3", "4", "5", "6", "7"),
    "8": UPCARDS,
    "9": ("2", "3", "4", "5", "6", "8", "9"),
}

# use these values to check that public hand totals are consistent
CARD_VALUES = {str(number): number for number in range(2, 11)}
CARD_VALUES.update({"Ace": 1, "Jack": 10, "Queen": 10, "King": 10})


# choose how to play blackjack hands using the public basic strategy chart
class BasicStrategy:
    def choose_action(self, observation):
        # check that it is this player's turn
        try:
            player_id = observation["player_id"]
            if observation["state"] != "player_turn" or observation["current_player"] != player_id:
                raise ValueError("It is not this player's turn")

            # this basic strategy chart is for six-deck blackjack
            round_starts = [event for event in observation["events"] if event["type"] == "round_start"]
            if len(round_starts) != 1 or round_starts[0]["num_decks"] != 6:
                raise ValueError("Basic Strategy requires six decks")

            # find the hand currently being played, including after splitting
            hands = observation["players"][player_id]["hands"]
            active_hands = [hand for hand in hands if hand["hand_id"] == observation["current_hand"]]
            if len(active_hands) != 1:
                raise ValueError("Cannot find the active hand")
            hand = active_hands[0]

            # check that the hand is ready to be played
            cards = hand["cards"]
            if hand["finished"] or hand["awaiting_card"] or len(cards) < 2:
                raise ValueError("The hand cannot be played")

            # check the publicly reported total and soft-hand status
            ranks = [card["value"] for card in cards]
            if any(rank not in CARD_VALUES for rank in ranks):
                raise ValueError("Invalid card")
            total = sum(CARD_VALUES[rank] for rank in ranks)
            soft = "Ace" in ranks and total + 10 <= 21
            if soft:
                total += 10
            if (type(hand["total"]) is not int or type(hand["soft"]) is not bool
                    or hand["total"] != total or hand["soft"] != soft or total >= 21):
                raise ValueError("Invalid hand total")

            # read the dealer's visible card, treating face cards as 10
            upcard = observation["dealer"]["cards"][0]["value"]
            if upcard in ("Jack", "Queen", "King"):
                upcard = "10"
            if upcard not in UPCARDS:
                raise ValueError("Invalid dealer upcard")
            column = UPCARDS.index(upcard)

            # only choose actions that the game engine permits
            legal_actions = observation["legal_actions"]
            if (not legal_actions or any(action not in ("hit", "stand", "double", "split")
                                         for action in legal_actions)):
                raise ValueError("Invalid legal actions")
            if "double" in legal_actions and len(cards) != 2:
                raise ValueError("Cannot double a hand with more than two cards")

            # split matching ranks only when splitting is legal and the chart recommends it
            if "split" in legal_actions and len(cards) == 2 and ranks[0] == ranks[1]:
                if upcard in SPLIT.get(ranks[0], ()):
                    return "split"

            # otherwise choose from the soft or hard hand chart
            if soft:
                choice = SOFT[total][column] if total <= 18 else "S"
            elif total < 9:
                choice = "H"
            elif total >= 17:
                choice = "S"
            else:
                choice = HARD[total][column]

            # use the correct fallback if doubling is not allowed
            if choice == "D":
                action = "double" if "double" in legal_actions else "hit"
            elif choice == "B":
                action = "double" if "double" in legal_actions else "stand"
            elif choice == "H":
                action = "hit"
            else:
                action = "stand"

            # never return an action that the engine says is illegal
            if action not in legal_actions:
                raise ValueError("The recommended action is unavailable")
            return action

        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise ValueError("Invalid playing observation") from error
