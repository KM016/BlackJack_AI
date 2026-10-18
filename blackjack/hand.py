class Hand:
    def __init__(self, cards=None, from_split=False):
        # store the cards in the hand
        self.cards = list(cards) if cards is not None else []

        # record whether the hand was created by splitting
        self.from_split = from_split

    def add_card(self, card):
        # add a new card to the hand
        self.cards.append(card)

    def _score(self):
        # start the hand total at zero
        total = 0
        aces = 0

        # calculate the value of each card
        for card in self.cards:
            value = card.value

            if value == "Ace":
                total += 1
                aces += 1

            elif value in ("Jack", "Queen", "King"):
                total += 10

            else:
                total += int(value)

        # count an Ace as 11 if this does not cause a bust
        soft = aces > 0 and total + 10 <= 21

        if soft:
            total += 10

        return total, soft

    def get_total(self):
        # get the bes (highest) possible total for the hand
        return self._score()[0]

    def is_soft(self):
        # check if an Ace is currently worth 11
        return self._score()[1]

    def is_blackjack(self):
        # natural blackjack must have exactly two cards
        # hands created by splitting cannot have a natural blackjack
        if len(self.cards) != 2 or self.from_split:
            return False

        values = [card.value for card in self.cards]

        # check for an Ace and a card worth 10
        ten_values = ("10", "Jack", "Queen", "King")

        return "Ace" in values and any(value in ten_values for value in values)

    def is_bust(self):
        # hand is bust if the total exceeds 21
        return self.get_total() > 21
