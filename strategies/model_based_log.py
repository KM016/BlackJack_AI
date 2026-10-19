
import math
from decimal import Decimal

from research.calibration_model import CalibrationModel, get_bin
from strategies.betting import betting_limits, decimal_value
from strategies.hi_lo import HiLoCounter
from strategies.wager_grid import legal_wager_mask, wager_for_action


# calculate expected log-bankroll growth for a wager using calibrated outcomes
def expected_log_gain(distribution, wager, bankroll):

    # convert the wager and bankroll into decimals for accurate calculations
    wager = decimal_value(wager, "wager")
    bankroll = decimal_value(bankroll, "bankroll")

    # reject negative wagers or an invalid bankroll
    if wager < 0 or bankroll <= 0:
        raise ValueError("Wager must be nonnegative and bankroll must be positive")

    # each outcome changes wealth by the wager times its per-unit return
    # the fraction tells us how much of our bankroll we are betting
    fraction = wager / bankroll

    # store the weighted log growth from every possible outcome
    terms = []
    total_rounds = 0

    # go through each possible return and how often it occurred in calibration
    for outcome, frequency in distribution:
        outcome = decimal_value(outcome, "return")

        # make sure the recorded outcome has a valid positive frequency
        if type(frequency) is not int or frequency <= 0:
            raise ValueError("Invalid return frequency")

        # calculate the change in bankroll as a fraction of current wealth
        # for example, a 2-unit bet on 100 units with a return of -1 loses 2%
        relative_profit = fraction * outcome

        # log growth is only defined when the remaining bankroll is positive
        if 1 + relative_profit <= 0:
            raise ValueError("Wager could leave no positive bankroll")

        # calculate log(1 + relative profit), weighted by how often it occurred
        # this penalises large losses and rewards sustainable bankroll growth
        terms.append(frequency * math.log1p(float(relative_profit)))
        total_rounds += frequency

    # we cannot calculate an expected value without any recorded outcomes
    if not total_rounds:
        raise ValueError("Cannot estimate log growth from an empty distribution")

    # divide the total weighted log growth by the number of observed rounds
    # this gives the expected log growth for the proposed wager
    return math.fsum(terms) / total_rounds


# choose the legal wager with the highest estimated log-bankroll growth
class ModelBasedLogBetting:

    def __init__(self, model, counter=None):

        # load an existing calibration model; never refit while making decisions
        if not isinstance(model, CalibrationModel):
            raise ValueError("Supply a fitted CalibrationModel")

        # get the saved calibration results and check they use the correct game setup
        document = model.get_document()
        if document.get("population") != "two_seats_fixed_one_unit_s17_das":
            raise ValueError("Model must use the two-seat fixed-unit calibration population")

        # get the number of decks used during calibration
        # create a Hi-Lo counter to track publicly revealed cards
        self.num_decks = document["metadata"]["rules"]["decks"]
        self.model = model
        self.counter = counter if counter is not None else HiLoCounter(num_decks=self.num_decks)

        # keep a record of our latest betting decision
        # store the calibrated return distributions so we can reuse them during play
        self.last_decision = None
        self.tables = []

        # save the small distribution of returns for each true-count bin
        # check the frequencies agree with the fitted means and variances
        for item in document["summary"]["bins"]:
            table = []
            used_returns = set()

            # read every possible return and how many times it occurred
            # for example, a return of -1 means losing the initial wager
            # a return of 1.5 means winning a natural blackjack
            for entry in item["distribution"]:
                value = decimal_value(entry["return"], "round return")
                frequency = entry["count"]

                # returns must be within the game's possible range
                # each return must appear once and have a positive integer frequency
                if (not -8 <= value <= 8 or value in used_returns
                        or type(frequency) is not int or frequency <= 0):
                    raise ValueError("Invalid calibrated return distribution")

                # remember the return and save its frequency
                used_returns.add(value)
                table.append((value, frequency))

            # the return frequencies must add up to the number of calibrated rounds
            count = sum(frequency for _, frequency in table)
            if count != item["count"]:
                raise ValueError("Return distribution count does not match calibration")

            # recalculate the average return and variance from the distribution
            # this verifies that the stored data agrees with our fitted model
            if count > 1:

                # calculate the total returns and squared returns
                # multiplying by frequency avoids repeating each individual observation
                total = sum((value * frequency for value, frequency in table), Decimal(0))
                squared = sum((value * value * frequency for value, frequency in table), Decimal(0))

                # calculate the mean return and sample variance
                mean = total / count
                variance = (squared - total * total / count) / (count - 1)

                # decimal rounding can differ slightly between two variance formulas
                # allow tiny rounding differences, but reject inconsistent statistics
                if (abs(mean - decimal_value(item["mean"], "mean")) > Decimal("1e-20")
                        or abs(variance - decimal_value(item["variance"], "variance")) > Decimal("1e-20")):
                    raise ValueError("Return distribution does not match fitted moments")

            # a statistically supported bin must contain enough data for variance
            elif item["supported"]:
                raise ValueError("Supported bin needs at least two returns")

            # save the checked distribution for this true-count bin
            self.tables.append(tuple(table))

    @classmethod
    def from_file(cls, path):
        # read the fitted model once, not on every betting decision
        return cls(CalibrationModel.load(path))

    def choose_wager(self, observation):
        # use only information available before the next round starts

        # get our current bankroll from the public betting observation
        bankroll, _, _, _ = betting_limits(observation)

        # make sure the number of decks matches the calibration model
        if observation["num_decks"] != self.num_decks:
            raise ValueError("Deck count differs from the calibrated game")

        # calculate the true count using publicly revealed cards
        # look up the estimated return statistics for that true-count range
        true_count = self.counter.get_betting_true_count(observation)
        prediction = self.model.predict(true_count)

        # find which wagers are allowed under our shared betting limits
        # the available actions are sitting out, 1, 2, 4, 6 and 8 units
        mask = legal_wager_mask(observation)

        # sitting out gives exactly zero log growth
        # we only choose a wager if it improves on doing nothing
        best_action = 0
        best_gain = 0.0
        reason = "no_supported_positive_edge"

        # only bet when the calibration interval supports a positive advantage
        # the lower confidence bound must be above zero
        if prediction is not None and prediction["mean_ci"][0] > 0:

            # get the full distribution of outcomes for our current true count
            table = self.tables[get_bin(true_count, self.model.boundaries)]

            # assume we will sit out unless a wager offers positive log growth
            reason = "no_positive_log_gain"

            # test each possible positive wager on our discrete betting grid
            for action in range(1, len(mask)):

                # skip wagers that are not legal or exceed our risk limits
                if not mask[action]:
                    continue

                # convert the action into a wager and calculate its expected log growth
                wager = wager_for_action(action)
                gain = expected_log_gain(table, wager, bankroll)

                # strict improvement means a smaller wager wins any tie
                # keep whichever legal bet produces the highest positive log growth
                if gain > best_gain:
                    best_action = action
                    best_gain = gain
                    reason = None

        # keep the decision available for analysis without changing the observation
        # this lets us inspect the true count, selected bet and expected log growth
        self.last_decision = {
            "true_count": true_count,
            "calibration_bin": prediction["bin"] if prediction is not None else None,
            "action_index": best_action,
            "estimated_log_gain": best_gain,
            "wager": wager_for_action(best_action),
            "sit_out_reason": reason
        }

        # return the chosen wager or None if sitting out is best
        return wager_for_action(best_action)
