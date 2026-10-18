from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


def money(value):
    # convert money into Decimal dtype to avoid floating point errors
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("Invalid monetary amount") from None

    # reject invalid amounts such as infinity or NaN
    if not amount.is_finite():
        raise ValueError("Invalid monetary amount")

    return amount


# store the rules used by the blackjack table
@dataclass(frozen=True)
class Rules:
    num_decks: int = 6
    penetration: float = 0.75
    min_bet: Decimal = Decimal("1")
    max_bet: Decimal = Decimal("1000")
    max_split_hands: int = 4

    def __post_init__(self):
        # check the number of decks
        if type(self.num_decks) is not int or self.num_decks < 1:
            raise ValueError("Invalid number of decks")

        # check the penetration limit
        if not 0 < self.penetration <= 1:
            raise ValueError("Invalid penetration")

        # check the maximum number of split hands
        if type(self.max_split_hands) is not int or not 2 <= self.max_split_hands <= 4:
            raise ValueError("Invalid split hand limit")

        # convert the betting limits into Decimals
        object.__setattr__(self, "min_bet", money(self.min_bet))
        object.__setattr__(self, "max_bet", money(self.max_bet))

        # check that the betting limits are valid
        if self.min_bet <= 0 or self.max_bet < self.min_bet:
            raise ValueError("Invalid betting limits")
