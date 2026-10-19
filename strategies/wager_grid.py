
from decimal import Decimal

from strategies.betting import betting_limits, decimal_value
from strategies.hi_lo import HiLoCounter


# use the same six betting choices for every strategy
# None represents sitting out, while the other values are betting amounts
# the bankroll limit restricts any initial wager to 2% of available money
WAGERS = (None, Decimal(1), Decimal(2), Decimal(4), Decimal(6), Decimal(8))
BANKROLL_LIMIT = Decimal("0.02")


# convert an action number into its wager (None means sit out)
# the optional unit lets DQN sessions scale all wagers together
def wager_for_action(action, unit=1):

    # make sure the action is an integer between 0 and 5
    # each number corresponds to a position in our WAGERS tuple
    if type(action) is not int or not 0 <= action < len(WAGERS):
        raise ValueError("Action must be an integer from 0 to 5")

    # convert the betting unit to a decimal and make sure it is positive
    unit = decimal_value(unit, "betting unit")
    if unit <= 0:
        raise ValueError("Betting unit must be positive")

    # multiply the selected wager by the unit size
    # for example, action 3 normally bets 4 units, or 8 if unit is 2
    # action 0 returns None because the player is sitting out
    return WAGERS[action] * unit if action else None


# check which wagers are allowed by the table and bankroll
# use the same rules if the betting unit is changed for a DQN session
def legal_wager_mask(observation, unit=1):

    # get the player's bankroll, table limits and whether betting is available
    bankroll, minimum, maximum, can_bet = betting_limits(observation)

    # make sure the betting unit is a positive decimal
    unit = decimal_value(unit, "betting unit")
    if unit <= 0:
        raise ValueError("Betting unit must be positive")

    # sitting out is always allowed between rounds
    # start a list of legal actions, where True means the action is allowed
    mask = [True]

    # check every positive wager in our betting grid
    for multiple in WAGERS[1:]:

        # calculate the actual stake using the chosen betting unit
        wager = multiple * unit

        # the initial wager is capped at 2% of bankroll
        # splitting four times and doubling can expose eight initial stakes
        # a wager is allowed only if:
        # betting is available, it satisfies the table limits,
        # it uses no more than 2% of bankroll,
        # and the worst possible exposure is strictly below our bankroll
        allowed = (can_bet and minimum <= wager <= maximum
                   and wager <= BANKROLL_LIMIT * bankroll
                   and 8 * wager < bankroll)

        # record whether this wager is legal
        mask.append(allowed)

    # return the legal actions in the same order as WAGERS
    # for example, (True, True, True, False, False, False)
    # means we can sit out, bet 1 or bet 2, but nothing larger
    return tuple(mask)


# select the largest legal grid wager that does not exceed the proposal
def proposed_action(proposed, mask):

    # make sure the mask contains exactly six True or False values
    if len(mask) != len(WAGERS) or any(type(allowed) is not bool for allowed in mask):
        raise ValueError("Invalid legal wager mask")

    # sitting out must always be available as an option
    if not mask[0]:
        raise ValueError("Sitting out must be a legal action")

    # if a strategy does not want to bet, select action 0
    if proposed is None:
        return 0

    # convert the strategy's proposed wager into a decimal
    # start with action 0 in case no positive wager is suitable
    amount = decimal_value(proposed, "proposed wager")
    action = 0

    # check every possible positive wager in increasing order
    for index in range(1, len(WAGERS)):

        # only select wagers that are legal and no larger than the proposal
        # because we go from smallest to largest, the final action is
        # the largest permitted wager that does not exceed the proposal
        if mask[index] and WAGERS[index] <= amount:
            action = index

    # return the action number, rather than the monetary wager itself
    return action


# apply the same discrete wager rules to any betting strategy
class DiscreteBetting:

    def __init__(self, policy=None):

        # None is the control strategy that never bets
        # store the original strategy so we can request its proposed wager
        self.policy = policy

        # remember which action was selected in the most recent round
        self.last_action = None

    def reset(self):
        # start a fresh independent session without carrying over a card count

        # clear the previous betting decision
        self.last_action = None

        # the never-bet strategy has nothing else to reset
        if self.policy is None:
            return

        # if the strategy has its own reset method, use it
        # this allows strategies to clear their internal state between sessions
        if callable(getattr(self.policy, "reset", None)):
            self.policy.reset()

        # otherwise, reset its Hi-Lo counter if it has one
        # this prevents card counts from one session affecting another
        elif hasattr(self.policy, "counter"):
            self.policy.counter = HiLoCounter(num_decks=self.policy.counter.num_decks)

    def choose_wager(self, observation):
        # let the strategy propose an amount before enforcing the common grid

        # ask the underlying strategy how much it wants to bet
        # the never-bet control always proposes None
        proposed = self.policy.choose_wager(observation) if self.policy is not None else None

        # check which wagers are legal and round the proposal down
        # to the largest permitted wager on our common betting grid
        self.last_action = proposed_action(proposed, legal_wager_mask(observation))

        # convert the selected action into a wager the engine understands
        # action 0 returns None, meaning the player sits out
        return wager_for_action(self.last_action)
