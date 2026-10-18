from copy import deepcopy

from blackjack.dealer import Dealer
from blackjack.history import History, card_record
from blackjack.observation import observe, observe_betting
from blackjack.player import Player
from blackjack.player_hand import PlayerHand
from blackjack.rules import Rules, money
from blackjack.shoe import Shoe


# signal when the shoe has run out of cards needed for the game to continue
class ShoeExhausted(RuntimeError):
    pass


# manage the blackjack game including dealing, turns, actions and payouts
class BlackjackEngine:
    def __init__(self, players=None, rules=None, seed=None, shoe=None, retain_history=True):
        # create the table with one player by default
        self.players = dict(players) if players is not None else {"player_1": Player()}

        # check that the table has between one and seven players
        if not 1 <= len(self.players) <= 7:
            raise ValueError("A table needs 1 to 7 players")

        # check that every player has a valid identifier and object
        for player_id, player in self.players.items():
            if not isinstance(player_id, str) or not player_id:
                raise ValueError("Player identifiers must be nonempty strings")
            if not isinstance(player, Player):
                raise ValueError("Each seat needs a Player")
            if player.bets:
                raise ValueError("Players must have no outstanding bets")

        # the same Player object cannot occupy multiple seats
        if len({id(player) for player in self.players.values()}) != len(self.players):
            raise ValueError("A Player cannot occupy two seats")

        # create the casino rules and shoe
        self.rules = rules if rules is not None else Rules()
        self.shoe = shoe if shoe is not None else Shoe(
            self.rules.num_decks, seed, self.rules.penetration
        )

        # check that the shoe matches the table rules
        if (self.shoe.num_decks != self.rules.num_decks
                or self.shoe.penetration_limit != self.rules.penetration):
            raise ValueError("Shoe settings must match the rules")

        # store completed rounds for later analysis
        self.history = History(retain_history)

        # initialise the state of the table
        self.round_number = 0
        self.state = "betting"
        self.dealer = Dealer()
        self.wagers = {}
        self.public_events = []
        self.current_player = None
        self.current_hand = None

        # track the players, hands and starting money in the current round
        self._seats = []
        self._next_hand = 0
        self._record = None

        for player in self.players.values():
            player.hands = []

    # create a unique identifier for each new hand
    def _hand_id(self):
        self._next_hand += 1
        return f"r{self.round_number}_h{self._next_hand}"

    # check that the shoe contains enough cards
    def _require_cards(self, count):
        if self.shoe.get_remaining() < count:
            raise ShoeExhausted("Not enough cards remaining in the shoe")

    # draw one card from the shoe
    def _draw(self):
        self._require_cards(1)
        return list(self.shoe.deal(1))[0]

    # record an event that players are allowed to observe
    def _event(self, kind, **details):
        self.public_events.append({"type": kind, **details})

    # check all wagers before deducting any money
    def _validate_wagers(self, wagers):
        if not wagers or any(player_id not in self.players for player_id in wagers):
            raise ValueError("Wagers must identify participating players")

        # reject a new round if any money is still tied up in an old bet
        if any(player.bets for player in self.players.values()):
            raise ValueError("Players have outstanding bets")

        validated = {player_id: money(amount) for player_id, amount in wagers.items()}

        for player_id, amount in validated.items():
            if not self.rules.min_bet <= amount <= self.rules.max_bet:
                raise ValueError("Wager is outside the table limits")
            if amount > self.players[player_id].bankroll:
                raise ValueError("Insufficient bankroll")

        return validated

    # start a round, place bets and deal the initial cards
    def start_round(self, wagers):
        # do not start a new round while another round is active
        if self.state not in ("betting", "complete", "void"):
            raise ValueError("A round is already active")

        # validate all wagers before changing the game
        validated = self._validate_wagers(wagers)

        # reshuffle between rounds if the penetration limit was reached
        reshuffled = self.shoe.needs_reshuffle()
        if reshuffled:
            self.shoe.reset()

        # each playing seat and the dealer need two cards
        self._require_cards(2 * (len(validated) + 1))

        # reset the state for the new round
        self.round_number += 1
        self._next_hand = 0
        self._seats = [player_id for player_id in self.players if player_id in validated]
        self.dealer = Dealer()
        self.wagers = {}
        self.public_events = []
        self.current_player = None
        self.current_hand = None
        self.state = "dealing"

        # save starting balances for results and possible refunds
        self._record = {
            "round_number": self.round_number,
            "shoe": {
                "shuffle_number": self.shoe.shuffle_count,
                "num_decks": self.shoe.num_decks,
                "penetration_start": self.shoe.get_penetration(),
                "remaining_start": self.shoe.get_remaining()
            },
            "players": {
                player_id: {
                    "bankroll_start": player.bankroll,
                    "initial_wager": validated.get(player_id, money(0))
                }
                for player_id, player in self.players.items()
            }
        }

        self._event("round_start", round_number=self.round_number,
                    seats=list(self._seats), num_decks=self.shoe.num_decks,
                    reshuffled=reshuffled)

        # create an initial hand and reserve each participating players bet
        for player in self.players.values():
            player.hands = []

        for player_id in self._seats:
            player = self.players[player_id]
            hand_id = self._hand_id()
            player.place_bet(hand_id, validated[player_id])
            self.wagers[hand_id] = player.bets[hand_id]
            player.hands = [PlayerHand(hand_id=hand_id)]
            self._event("bet", player_id=player_id, hand_id=hand_id,
                        amount=validated[player_id])

        # deal initial cards and check for natural blackjacks
        self._deal_initial_cards()
        if not self._check_initial_blackjacks():
            self._advance()

    # deal each players first card, then the dealers, twice
    def _deal_initial_cards(self):
        for pass_number in range(2):
            for player_id in self._seats:
                hand = self.players[player_id].hands[0]
                card = self._draw()
                hand.hand.add_card(card)
                self._event("card", player_id=player_id,
                            hand_id=hand.hand_id, card=card_record(card))

            card = self._draw()
            self.dealer.hand.add_card(card)

            # only the first dealer card is visible initially
            if pass_number == 0:
                self._event("dealer_upcard", card=card_record(card))

        # reaching 21 on the initial deal automatically finishes the hand
        for player_id in self._seats:
            hand = self.players[player_id].hands[0]
            hand.finished = hand.hand.get_total() >= 21

    # settle initial blackjacks before any player actions
    def _check_initial_blackjacks(self):
        # dealer blackjack ends the round immediately
        if self.dealer.peek():
            self._event("dealer_reveal", card=card_record(self.dealer.hand.cards[1]))

            for player_id in self._seats:
                hand = self.players[player_id].hands[0]
                outcome = "push" if hand.hand.is_blackjack() else "loss"
                self._settle(player_id, hand, outcome)

            self._complete()
            return True

        # pay any player natural blackjack even if other players must still act
        for player_id in self._seats:
            hand = self.players[player_id].hands[0]
            if hand.hand.is_blackjack():
                self._settle(player_id, hand, "blackjack")

        return False

    # return the actions currently available to a player
    def legal_actions(self, player_id):
        if self.state != "player_turn" or player_id != self.current_player:
            return []

        hand = self.current_hand
        if hand is None or hand.finished or hand.awaiting_card:
            return []

        actions = ["hit", "stand"]
        player = self.players[player_id]

        # doubling and splitting require another wager of the same size
        can_afford = player.bankroll >= self.wagers[hand.hand_id]

        if hand.can_double() and can_afford:
            actions.append("double")

        if (hand.can_split() and can_afford
                and len(player.hands) < self.rules.max_split_hands):
            actions.append("split")

        return actions

    # carry out a players chosen action
    def act(self, player_id, action):
        # reject illegal actions without changing the round
        if action not in self.legal_actions(player_id):
            raise ValueError("Illegal action or wrong player turn")

        hand = self.current_hand

        # check the next card before making any changes
        self._require_cards(0 if action == "stand" else 1)

        additional_wager = (self.wagers[hand.hand_id]
                            if action in ("double", "split") else money(0))
        new_hand_id = self._hand_id() if action == "split" else None

        self._event("action", player_id=player_id, hand_id=hand.hand_id,
                    action=action, additional_wager=additional_wager,
                    new_hand_id=new_hand_id)

        if action == "stand":
            hand.stand()

        elif action == "hit":
            card = self._draw()
            hand.hit(card)
            self._event("card", player_id=player_id,
                        hand_id=hand.hand_id, card=card_record(card))

        elif action == "double":
            self._double_hand(player_id, hand)

        elif action == "split":
            self._split_hand(player_id, hand, new_hand_id)

        # move to the next active hand or finish the round
        self._advance()

    # double a wager, give the player one card and finish the hand
    def _double_hand(self, player_id, hand):
        player = self.players[player_id]
        player.increase_bet(hand.hand_id)
        self.wagers[hand.hand_id] = player.bets[hand.hand_id]

        card = self._draw()
        hand.double_down(card)
        self._event("card", player_id=player_id,
                    hand_id=hand.hand_id, card=card_record(card))

    # split a matching pair into two separately wagered hands
    def _split_hand(self, player_id, hand, new_hand_id):
        player = self.players[player_id]

        # reserve another bet equal to the original hand's current wager
        player.place_bet(new_hand_id, self.wagers[hand.hand_id])
        self.wagers[new_hand_id] = player.bets[new_hand_id]

        # remember whether the player split Aces
        split_aces = hand.hand.cards[0].value == "Ace"

        # each new hand starts with one card and awaits its replacement
        first = PlayerHand([hand.hand.cards[0]], from_split=True,
                           hand_id=hand.hand_id, split_aces=split_aces)
        second = PlayerHand([hand.hand.cards[1]], from_split=True,
                            hand_id=new_hand_id, split_aces=split_aces)

        # replace the original hand, keeping resplits in the correct turn order
        index = player.hands.index(hand)
        player.hands[index:index + 1] = [first, second]

    # find the next playable hand or finish the round
    def _advance(self):
        for player_id in self._seats:
            for hand in self.players[player_id].hands:
                # deal the replacement only when a split hand's turn arrives
                if hand.awaiting_card:
                    try:
                        card = self._draw()
                    except ShoeExhausted:
                        self._void_round()
                        raise

                    hand.receive_split_card(card)
                    self._event("card", player_id=player_id,
                                hand_id=hand.hand_id, card=card_record(card))

                # settle busted hands immediately
                if hand.hand.is_bust() and hand.outcome is None:
                    self._settle(player_id, hand, "loss")

                # stop at the first hand that can still act
                if not hand.finished:
                    self.state = "player_turn"
                    self.current_player = player_id
                    self.current_hand = hand
                    return

        # all player hands are finished, so the dealer can now play
        self.current_player = None
        self.current_hand = None
        self.state = "dealer_turn"
        self._play_dealer()

    # settle a hands wager and store its result
    def _settle(self, player_id, hand, outcome):
        if hand.outcome is not None:
            raise ValueError("Hand is already settled")

        hand.net_profit = self.players[player_id].settle_bet(hand.hand_id, outcome)
        hand.outcome = outcome
        hand.finished = True

        self._event("settlement", player_id=player_id,
                    hand_id=hand.hand_id, outcome=outcome,
                    wager=self.wagers[hand.hand_id], net_profit=hand.net_profit)

    # reveal the dealers cards and settle all remaining hands
    def _play_dealer(self):
        self.dealer.reveal()
        self._event("dealer_reveal", card=card_record(self.dealer.hand.cards[1]))

        # only hands that have not already settled need the dealer to play
        unsettled = []
        for player_id in self._seats:
            for hand in self.players[player_id].hands:
                if hand.outcome is None:
                    unsettled.append((player_id, hand))

        try:
            if unsettled:
                # record each additional dealer card when it is drawn
                def draw():
                    card = self._draw()
                    self._event("dealer_card", card=card_record(card))
                    return card

                self.dealer.play(draw)
        except ShoeExhausted:
            self._void_round()
            raise

        # compare each hand against the dealer independently
        for player_id, hand in unsettled:
            player_total = hand.hand.get_total()
            dealer_total = self.dealer.hand.get_total()

            if self.dealer.hand.is_bust() or player_total > dealer_total:
                outcome = "win"
            elif player_total == dealer_total:
                outcome = "push"
            else:
                outcome = "loss"

            self._settle(player_id, hand, outcome)

        self._complete()

    # refund every wager if a compulsory draw cannot be completed
    def _void_round(self):
        for player_id, player in self.players.items():
            # restore the balance from before the round
            player.bankroll = self._record["players"][player_id]["bankroll_start"]
            player.bets.clear()

            for hand in player.hands:
                hand.finished = True
                hand.awaiting_card = False
                hand.outcome = "void"
                hand.net_profit = money(0)

        # reveal the dealers hidden card if it is still hidden
        if not self.dealer.revealed:
            self.dealer.reveal()
            self._event("dealer_reveal", card=card_record(self.dealer.hand.cards[1]))

        self._event("round_void", reason="shoe_exhausted")
        self._complete("void")

    # save the completed round and its results
    def _complete(self, state="complete"):
        self.state = state
        self.current_player = None
        self.current_hand = None
        self._event("round_end", state=state)

        # copy the starting record before adding the final results
        record = deepcopy(self._record)
        record["state"] = state
        record["dealer_cards"] = [card_record(card) for card in self.dealer.hand.cards]
        record["events"] = deepcopy(self.public_events)
        record["shoe"]["remaining_end"] = self.shoe.get_remaining()
        record["shoe"]["penetration_end"] = self.shoe.get_penetration()

        # record every players hands, wagers and bankroll change
        for player_id, player in self.players.items():
            hands = []

            for hand in player.hands:
                hands.append({
                    "hand_id": hand.hand_id,
                    "cards": [card_record(card) for card in hand.hand.cards],
                    "from_split": hand.hand.from_split,
                    "split_aces": hand.split_aces,
                    "doubled": hand.doubled,
                    "wager": self.wagers[hand.hand_id],
                    "outcome": hand.outcome,
                    "net_profit": hand.net_profit
                })

            record["players"][player_id].update({
                "bankroll_end": player.bankroll,
                "net_profit": player.bankroll - record["players"][player_id]["bankroll_start"],
                "hands": hands
            })

        self.history.append(record)

    # return information visible to a player during a round
    def observation(self, player_id):
        return observe(self, player_id)

    # return information visible before placing the next bet
    def betting_observation(self, player_id):
        return observe_betting(self, player_id)
