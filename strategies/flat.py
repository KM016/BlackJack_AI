from strategies.betting import decimal_value, betting_limits


# place the same wager every round, provided the player can afford it
class FlatBetting:
    def __init__(self, stake=10):
        # store the fixed wager as an exact Decimal
        self.stake = decimal_value(stake, "stake")

        # the fixed wager must be positive
        if self.stake <= 0:
            raise ValueError("Stake must be positive")

    # choose a wager before the next round starts
    def choose_wager(self, observation):
        # read the player's bankroll and the table's betting limits
        bankroll, minimum, maximum, can_bet = betting_limits(observation)

        # reject a configured wager outside the table limits
        if not minimum <= self.stake <= maximum:
            raise ValueError("Fixed stake is outside the table limits")

        # sit out when the player cannot afford the fixed wager
        if not can_bet or bankroll < self.stake:
            return None

        # always bet the original fixed amount
        return self.stake
