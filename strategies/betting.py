from decimal import Decimal, InvalidOperation

# helper file that makes sure the betting strategies work with valid amounts of money and stay within betting limits

# convert input into an exact Decimal for betting calculations
def decimal_value(value, name):
    # accept integers, strings and Decimals, but not floats or booleans
    if type(value) not in (int, str, Decimal):
        raise ValueError(f"{name} must be an integer, string or Decimal")

    # convert the value into a Decimal
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(f"Invalid {name}") from None

    # reject values such as NaN and infinity
    if not amount.is_finite():
        raise ValueError(f"Invalid {name}")

    return amount


# get the bankroll (total amount player has to gamble with) and betting limits from a public observation
def betting_limits(observation):
    try:
        # betting decisions must be made before a round starts
        if observation["state"] not in ("betting", "complete", "void"):
            raise ValueError("Betting is only allowed between rounds")

        # read the players bankroll and the tables betting limits
        bankroll = decimal_value(observation["bankroll"], "bankroll")
        minimum = decimal_value(observation["min_bet"], "minimum bet")
        maximum = decimal_value(observation["max_bet"], "maximum bet")
        can_bet = observation["can_bet"]

        # check that the public betting information makes sense
        if bankroll < 0 or minimum <= 0 or maximum < minimum or type(can_bet) is not bool:
            raise ValueError("Invalid betting limits")

        return bankroll, minimum, maximum, can_bet

    except (KeyError, TypeError) as error:
        raise ValueError("Invalid betting observation") from error
