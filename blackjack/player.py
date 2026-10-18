from decimal import Decimal
from blackjack.rules import money

# represents players money and actions, not their strategy
class Player:
    def __init__(self, bankroll=1000):
        # store the players starting bankroll
        self.bankroll = money(bankroll)

        # check the bankroll is valid
        if self.bankroll < 0:
            raise ValueError("Invalid starting bankroll")

        # store active bets for each hand
        self.bets = {}

        # store the players hands for the current round
        self.hands = []

    def place_bet(self, hand_id, amount):
        # convert the bet into a Decimal
        amount = money(amount)

        # check the bet is positive
        if amount <= 0:
            raise ValueError("Bet must be positive")

        # check the player has enough money
        if amount > self.bankroll:
            raise ValueError("Insufficient bankroll")

        # check that this hand does not already have a bet
        if hand_id in self.bets:
            raise ValueError("Hand already has a bet")

        # subtract the bet from the available bankroll
        self.bankroll -= amount

        # record the active bet
        self.bets[hand_id] = amount

    def increase_bet(self, hand_id):
        # check that the hand has an active bet
        if hand_id not in self.bets:
            raise ValueError("No active bet for this hand")

        # doubling requires another stake equal to the current bet
        amount = self.bets[hand_id]

        # check that the player can afford the additional bet
        if amount > self.bankroll:
            raise ValueError("Insufficient bankroll")

        # subtract the additional stake from the bankroll
        self.bankroll -= amount

        # double the amount wagered on this hand
        self.bets[hand_id] += amount

    def settle_bet(self, hand_id, outcome):
        # define the total payout for each possible outcome
        # multipliers include returning the original stake
        payouts = {
            "win": Decimal("2"),
            "loss": Decimal("0"),
            "push": Decimal("1"),
            "blackjack": Decimal("2.5")
        }

        # check that the outcome is valid
        if outcome not in payouts:
            raise ValueError("Invalid outcome")

        # check that the hand has an active bet
        if hand_id not in self.bets:
            raise ValueError("No active bet for this hand")

        # remove the settled bet from the active bets
        wager = self.bets.pop(hand_id)

        # calculate the money returned to the player
        returned = wager * payouts[outcome]

        # add the returned money to the bankroll
        self.bankroll += returned

        # return the net profit or loss
        return returned - wager
