from blackjack.hand import Hand

# manage a players individual hand: actions and splitting rules
class PlayerHand:
    def __init__(self, cards=None, from_split=False, hand_id=None, split_aces=False):
        # create a hand using the Hand class
        self.hand = Hand(cards, from_split=from_split)

        # store the hands identifier
        self.hand_id = hand_id

        # track the actions taken on this hand
        self.has_hit = False
        self.doubled = False

        # track whether this hand was created by splitting Aces
        self.split_aces = split_aces

        # split hands start with one card and wait for a replacement
        self.awaiting_card = from_split and len(self.hand.cards) == 1

        # a hand is finished if it already has 21 or more
        self.finished = self.hand.get_total() >= 21

        # split Aces cannot take additional cards after their replacement
        if self.split_aces and not self.awaiting_card:
            self.finished = True

        # store the result once the hand has been settled
        self.outcome = None
        self.net_profit = None

    def receive_split_card(self, card):
        # check that the hand is waiting for a replacement card
        if not self.awaiting_card or self.finished:
            raise ValueError("Hand is not waiting for a split card")

        # add the replacement card to the hand
        self.hand.add_card(card)
        self.awaiting_card = False

        # finish split Aces or any hand that has reached 21
        self.finished = self.split_aces or self.hand.get_total() >= 21

    def can_double(self):
        # doubling is only allowed before hitting and with two cards
        return (
            not self.finished
            and not self.awaiting_card
            and not self.has_hit
            and len(self.hand.cards) == 2
        )

    def can_split(self):
        # splitting needs two cards of the same rank
        # split Aces cannot be split again
        return (
            self.can_double()
            and not self.split_aces
            and self.hand.cards[0].value == self.hand.cards[1].value
        )

    def hit(self, card):
        # check that the hand is still active
        if self.finished:
            raise ValueError("Hand is already finished")

        # replacement cards after splitting are handled separately
        if self.awaiting_card:
            raise ValueError("Hand is waiting for a split card")

        # add the card and record that the player has hit
        self.hand.add_card(card)
        self.has_hit = True

        # finish the hand if the total reaches or exceeds 21
        if self.hand.get_total() >= 21:
            self.finished = True

    def stand(self):
        # check that the hand is still active
        if self.finished:
            raise ValueError("Hand is already finished")

        # check that the hand is not waiting for a split card
        if self.awaiting_card:
            raise ValueError("Hand is waiting for a split card")

        # finish the player's turn without drawing another card
        self.finished = True

    def double_down(self, card):
        # check that doubling is allowed
        if not self.can_double():
            raise ValueError("Hand cannot be doubled")

        # doubling allows exactly one additional card
        self.hit(card)

        # record the double and finish the hand
        self.doubled = True
        self.finished = True
