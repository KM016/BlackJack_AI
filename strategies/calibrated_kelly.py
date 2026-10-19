
from decimal import Decimal

from research.calibration_model import CalibrationModel
from strategies.betting import betting_limits, decimal_value
from strategies.hi_lo import HiLoCounter


# size blackjack bets using the empirical mean and variance from calibration
class CalibratedKellyBetting:

    def __init__(self, model, kelly_multiplier="0.5", max_bankroll_fraction="0.02",
                 max_exposure_safe_fraction="0.12", counter=None):

        # require a valid fitted model, not estimates invented during play
        if not isinstance(model, CalibrationModel):
            raise ValueError("Supply a fitted CalibrationModel")

        # store the calibration model and set our betting risk limits
        # half Kelly uses 50% of the full Kelly fraction to reduce risk
        # we also limit the initial bet to 2% of our current bankroll
        self.model = model
        self.kelly_multiplier = self._positive_fraction(kelly_multiplier, "kelly_multiplier")
        self.max_bankroll_fraction = self._positive_fraction(max_bankroll_fraction, "max_bankroll_fraction")
        self.max_exposure_safe_fraction = self._positive_fraction(
            max_exposure_safe_fraction, "max_exposure_safe_fraction"
        )

        # four hands can each double: at most eight initial stakes may be at risk
        if self.max_exposure_safe_fraction >= Decimal(1) / Decimal(8):
            raise ValueError("Exposure safety fraction must be strictly below 1/8")

        # use the same number of decks as the calibration simulations
        # initialise a Hi-Lo counter to track cards revealed during play
        self.num_decks = model.get_document()["metadata"]["rules"]["decks"]
        self.counter = counter if counter is not None else HiLoCounter(num_decks=self.num_decks)

        # store information about the most recent betting decision
        self.last_decision = None

    @staticmethod
    def _positive_fraction(value, name):
        # use exact monetary decimals and reject invalid fractions
        number = decimal_value(value, name)
        if not 0 < number <= 1:
            raise ValueError(f"{name} must be greater than 0 and at most 1")
        return number

    @classmethod
    def from_file(cls, path, **settings):
        # load the model once; do not refit during a simulated round
        return cls(CalibrationModel.load(path), **settings)

    def choose_wager(self, observation):
        # use only the public pre-round observation and the frozen model

        # get our current bankroll, table betting limits and whether we can bet
        bankroll, minimum, maximum, can_bet = betting_limits(observation)

        # make sure the game uses the same number of decks as our model
        if observation["num_decks"] != self.num_decks:
            raise ValueError("Deck count differs from the calibrated game")

        # calculate the true count using publicly revealed cards
        # look up the corresponding expected return and risk from calibration
        true_count = self.counter.get_betting_true_count(observation)
        prediction = self.model.predict(true_count)

        # if the true-count bin has insufficient reliable data, sit out
        if prediction is None:
            self.last_decision = {"true_count": true_count, "wager": None,
                                  "sit_out_reason": "unsupported_bin"}
            return None

        # extract the estimated average return, variance and confidence interval
        # mean tells us our expected profit per unit wagered
        # variance tells us how much the possible returns fluctuate
        # the lower confidence bound helps determine if our edge is reliable
        mean = prediction["mean"]
        variance = prediction["variance"]
        lower_confidence_bound = prediction["mean_ci"][0]

        # full Kelly is approximated by expected return divided by its variance
        # higher expected returns increase the fraction, while higher variance reduces it
        full_fraction = max(Decimal(0), mean / variance)

        # apply our Kelly multiplier, which defaults to half Kelly (50%)
        half_kelly_fraction = self.kelly_multiplier * full_fraction

        # cap the initial stake to control the risk from splits and doubles
        # choose the smallest fraction allowed by Kelly and our risk limits
        fraction = min(half_kelly_fraction, self.max_bankroll_fraction,
                       self.max_exposure_safe_fraction)

        # convert the chosen bankroll fraction into an actual wager
        # the wager cannot exceed the table maximum or available bankroll
        proposed_wager = bankroll * fraction
        wager = min(proposed_wager, maximum, bankroll)

        # decide whether we have a valid reason to sit out instead of betting
        reason = None

        # sit out if the player cannot afford the table's minimum wager
        if not can_bet or bankroll < minimum:
            reason = "insufficient_bankroll"

        # sit out if our calibration model predicts no positive expected return
        elif mean <= 0:
            reason = "no_estimated_edge"

        # sit out if the confidence interval includes zero or negative returns
        # a positive average alone is not enough evidence to justify betting
        elif lower_confidence_bound <= 0:
            reason = "uncertain_positive_edge"

        # sit out if our calculated wager is smaller than the table minimum
        elif wager < minimum:
            # do not round up: that would exceed the chosen risk allocation
            reason = "below_table_minimum"

        # if any of these conditions apply, do not place a wager
        if reason is not None:
            wager = None

        # save the complete decision so we can inspect why a bet was made
        # or why the strategy decided to sit out
        self.last_decision = {
            "true_count": true_count, "calibration_bin": prediction["bin"],
            "estimated_edge": mean, "return_variance": variance,
            "mean_ci": prediction["mean_ci"], "full_kelly_approx": full_fraction,
            "ideal_fraction": half_kelly_fraction, "chosen_fraction": fraction,
            "proposed_wager": proposed_wager, "wager": wager,

            # record whether our risk limits reduced the original Kelly wager
            "capped": fraction < half_kelly_fraction or (
                wager is not None and wager < proposed_wager
            ),

            # record why we sat out, or None if we placed a bet
            "sit_out_reason": reason
        }

        # return the final wager, or None if we decided to sit out
        return wager
