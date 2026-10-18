from blackjack.hand import Hand


class Dealer:
    def __init__(self, cards=None):
        # create the dealers hand
        self.hand = Hand(cards)

        # track whether the dealers hidden card has been revealed
        self.revealed = False

    def get_upcard(self):
        # return the dealers first (visible) card
        if self.hand.cards:
            return self.hand.cards[0]

        return None

    def peek(self):
        # check if the dealer has a visible card
        upcard = self.get_upcard()

        if upcard is None:
            return False

        # blackjack is can only happen with an Ace or ten value upcard
        ten_values = ("10", "Jack", "Queen", "King")

        if upcard.value not in ("Ace", *ten_values):
            return False

        # check if the dealer has a natural blackjack
        blackjack = self.hand.is_blackjack()

        # reveal the hidden card if the dealer has blackjack
        if blackjack:
            self.revealed = True

        return blackjack

    def reveal(self):
        # reveal the dealers hidden card
        self.revealed = True

    def should_hit(self):
        # the dealer must hit below 17 and stand on 17 or above
        return self.hand.get_total() < 17

    def play(self, draw):
        # reveal the hidden card before the dealer plays
        self.reveal()

        # keep drawing cards until the dealer reaches 17 or higher
        while self.should_hit():
            self.hand.add_card(draw())
