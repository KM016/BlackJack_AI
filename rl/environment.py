
import math
from copy import deepcopy
from decimal import Decimal

from blackjack.engine import BlackjackEngine, ShoeExhausted
from blackjack.player import Player
from blackjack.rules import Rules
from strategies.basic import BasicStrategy
from strategies.betting import betting_limits, decimal_value
from strategies.hi_lo import HiLoCounter
from strategies.wager_grid import legal_wager_mask, wager_for_action


# define the six public values that the DQN will use to choose a bet
# these describe the cards, our bankroll and how much of the session is left
FEATURE_NAMES = (
    "true_count", "running_count", "decks_remaining", "relative_bankroll",
    "drawdown", "rounds_remaining"
)


# convert a public betting observation into six small numerical features
# use this same function during training and when the trained DQN plays
# no private cards, shoe order or future outcomes are available here
def public_state(observation, counter, initial_bankroll, peak_bankroll, steps, rounds, unit=1):

    # get the current bankroll and calculate the true count using Hi-Lo
    bankroll, _, _, _ = betting_limits(observation)
    true_count = counter.get_betting_true_count(observation)

    # a pending shuffle means the next round will begin with a neutral count
    # reset the observed count and estimated remaining cards for the new shoe
    if observation["reshuffle_pending"]:
        running_count = 0
        remaining_fraction = Decimal(1)

    # otherwise use the cards revealed so far to estimate how much of the shoe remains
    else:
        running_count = counter.running_count
        remaining_cards = 52 * counter.num_decks - counter.cards_seen
        remaining_fraction = Decimal(remaining_cards) / (52 * counter.num_decks)

    # scale the values so the neural network receives manageable numbers
    # counts are clipped only for the network input, not in the Hi-Lo counter

    # feature 1: true count scaled to roughly -1 to 1
    # limit extreme true counts to prevent unusually large network inputs

    # feature 2: running count divided by 52 and limited to -1 to 1

    # feature 3: fraction of cards remaining in the shoe
    # 1 means a full shoe, while smaller values mean more cards have been dealt

    # feature 4: current bankroll relative to starting bankroll
    # 1 means unchanged, 2 means doubled, and values are capped at 4

    # feature 5: drawdown from the highest bankroll reached so far
    # 0 means we are at our peak, while larger values represent greater losses

    # feature 6: proportion of the session remaining
    # starts at 1 and approaches 0 as we complete rounds
    features = (
        float(max(Decimal(-10), min(Decimal(10), true_count)) / 10),
        max(-52, min(52, running_count)) / 52,
        float(remaining_fraction),
        float(min(Decimal(4), bankroll / initial_bankroll)),
        float(1 - bankroll / peak_bankroll),
        max(0, rounds - steps) / rounds,
    )

    # reuse exactly the same wager restrictions as the other strategies
    # the mask tells the DQN which of its six actions are currently allowed
    mask = legal_wager_mask(observation, unit)

    # return the features and legal actions together as one observation
    return {"features": features, "legal_mask": mask}


# provide real blackjack rounds for the DQN to learn betting decisions
class BlackjackBettingEnv:

    def __init__(self, initial_bankroll="1000", rounds=500, unit="1", rules=None):

        # the reference player always bets while the agent may choose to sit out

        # set the agent's starting bankroll and the value of one betting unit
        self.initial_bankroll = decimal_value(initial_bankroll, "initial bankroll")
        self.unit = decimal_value(unit, "betting unit")

        # make sure the starting bankroll and betting unit are valid
        if self.initial_bankroll <= 0 or self.unit <= 0:
            raise ValueError("Bankroll and betting unit must be positive")

        # each training episode must contain at least one round
        if type(rounds) is not int or rounds < 1:
            raise ValueError("Rounds must be a positive integer")

        # store the number of rounds and the blackjack rules
        # use our standard rules unless different rules were supplied
        self.rounds = rounds
        self.rules = rules if rules is not None else Rules()

        # keep the same six-deck blackjack rules used throughout the project
        # reject games that would differ from our existing research setup
        if (self.rules.num_decks != 6 or self.rules.penetration != 0.75
                or self.rules.max_split_hands != 4):
            raise ValueError("DQN environment requires six decks, 75% penetration and four split hands")

        # the reference player's fixed wager must be allowed by the table
        if not self.rules.min_bet <= self.unit <= self.rules.max_bet:
            raise ValueError("One reference betting unit must fit the table limits")

        # an episode starts when reset() creates a fresh seeded game
        # no engine exists until the first reset
        self._engine = None

        # track whether the episode has finished and how many rounds have been played
        self.done = True
        self.steps = 0
        self.terminal_reason = None

    # begin a new independent session with a fresh shoe and public card count
    def reset(self, seed):

        # each session uses a valid random seed so results are reproducible
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")

        # keep the reference seat first, so the dealing order is reproducible
        # the reference has a very large bankroll so it can keep revealing cards
        # the agent starts with the smaller bankroll we want the DQN to manage
        self._engine = BlackjackEngine({
            "reference": Player("1000000000"),
            "agent": Player(self.initial_bankroll),
        }, rules=self.rules, seed=seed, retain_history=False)

        # reset the Hi-Lo counter so no card information carries between sessions
        self.counter = HiLoCounter(num_decks=self.rules.num_decks)

        # use Basic Strategy to play hands once wagers have been chosen
        # the DQN only learns betting decisions, not how to hit or stand
        self.playing = BasicStrategy()

        # restart the round counter and highest recorded bankroll
        self.steps = 0
        self.peak_bankroll = self.initial_bankroll

        # prepare the features and legal wagers for the first round
        return self._next_state()

    # build the next observation using only information visible before betting
    def _next_state(self):

        # get the agent's public betting information from the engine
        observation = self._engine.betting_observation("agent")

        # update the highest bankroll reached during this episode
        # we use this later to calculate the agent's current drawdown
        self.peak_bankroll = max(self.peak_bankroll, observation["bankroll"])

        # convert the game information into six DQN features and legal actions
        self.current_state = public_state(
            observation, self.counter, self.initial_bankroll,
            self.peak_bankroll, self.steps, self.rounds, self.unit
        )

        # end the episode at its horizon or if no affordable bet remains
        # reaching the horizon means the required number of rounds is finished
        # having no positive legal wager means the agent cannot continue betting
        mask = self.current_state["legal_mask"]
        self.done = self.steps >= self.rounds or not any(mask[1:])

        # record why the episode ended, if it has finished
        self.terminal_reason = None
        if self.done:
            self.terminal_reason = "horizon" if self.steps >= self.rounds else "no_positive_wager"

            # disable every action so the DQN cannot act after termination
            self.current_state["legal_mask"] = (False,) * 6

        # give the learner the updated public observation
        return self.get_observation()

    # give the learner its own copy, not access to internal blackjack objects
    def get_observation(self):

        # the environment must be initialised before an observation can be requested
        if self._engine is None:
            raise ValueError("Reset the environment before requesting an observation")

        # copy the features and action mask so external code cannot change our state
        return deepcopy(self.current_state)

    # take one betting action, play one round and return the observed reward
    def step(self, action):

        # reject actions if no session has started or the episode has finished
        if self._engine is None or self.done:
            raise ValueError("Reset the environment before taking an action")

        # the DQN must choose one of our six action numbers: 0 to 5
        if type(action) is not int or not 0 <= action < 6:
            raise ValueError("Action must be an integer from 0 to 5")

        # check that the chosen wager is affordable and respects our risk limits
        if not self.current_state["legal_mask"][action]:
            raise ValueError("That wager is not legal for the current bankroll")

        # action zero is sitting out; the reference still plays to reveal cards

        # convert the chosen action into its wager
        # action zero becomes None because the agent is not participating
        wager = wager_for_action(action, self.unit)

        # record both bankrolls before the round so we can check the results
        bankroll_before = self._engine.betting_observation("agent")["bankroll"]
        reference_before = self._engine.betting_observation("reference")["bankroll"]

        # ensure the reference player can afford every possible split or double
        if reference_before < 8 * self.unit:
            raise RuntimeError("Reference bankroll cannot fund all legal actions")

        # the reference player always bets one unit
        # only add the agent if the DQN selected a positive wager
        wagers = {"reference": self.unit}
        if wager is not None:
            wagers["agent"] = wager

        # use the existing game engine and Basic Strategy for every active hand
        # remember the previous round number to check for valid void rounds
        previous_round = self._engine.round_number

        try:
            # start the blackjack round using the selected wagers
            self._engine.start_round(wagers)

            # automatically play every hand until no player actions remain
            while self._engine.state == "player_turn":

                # identify whose turn it is and choose the Basic Strategy move
                seat = self._engine.current_player
                move = self.playing.choose_action(self._engine.observation(seat))

                # apply the chosen hit, stand, double or split action
                self._engine.act(seat, move)

        except ShoeExhausted:
            # a genuine void round is refunded; other shoe errors must surface
            if self._engine.state != "void" or self._engine.round_number == previous_round:
                raise

        # make sure the round was settled or properly voided
        if self._engine.state not in ("complete", "void"):
            raise RuntimeError("Blackjack round did not finish")

        # calculate reward only after the cards have been played and settled

        # get the round's outcome and calculate the agent's actual profit or loss
        status = self._engine.state
        bankroll_after = self._engine.betting_observation("agent")["bankroll"]
        profit = bankroll_after - bankroll_before

        # our wagering limits should never allow the agent's bankroll to reach zero
        if bankroll_after <= 0:
            raise RuntimeError("Permitted wager left no positive bankroll")

        # cancelled rounds must refund both players completely
        # because no money was won or lost, the DQN receives zero reward
        if status == "void":
            if profit != 0 or self._engine.betting_observation("reference")["bankroll"] != reference_before:
                raise RuntimeError("Voided round did not refund every wager")
            reward = 0.0

        # sitting out must leave the agent's bankroll unchanged
        # the reference still plays, but the agent receives zero reward
        elif wager is None:
            if profit != 0:
                raise RuntimeError("Sitting out must not change the agent bankroll")
            reward = 0.0

        # calculate the reward from the actual bankroll change
        else:
            # log(after / before) is the change in log bankroll from this round
            # positive rewards mean the bankroll increased
            # negative rewards mean it decreased
            # larger relative losses receive a stronger penalty
            reward = math.log1p(float(profit / bankroll_before))

        # advance the public card counter and prepare the next legal actions
        # increase the number of completed round opportunities in this episode
        self.steps += 1

        # calculate the updated features, action mask and termination status
        next_state = self._next_state()

        # give training code round results without exposing any hidden cards

        # get the public betting view for the upcoming round
        # calculate the true count after the previous round's cards were revealed
        next_betting_view = self._engine.betting_observation("agent")
        upcoming_count = self.counter.get_betting_true_count(next_betting_view)

        # store useful results so we can analyse what happened during training
        # these include the chosen bet, profit, true count and termination reason
        info = {
            "action_index": action,
            "wager": wager,
            "bankroll_before": bankroll_before,
            "bankroll_after": bankroll_after,
            "net_profit": profit,

            # the return is profit divided by the initial wager
            # sitting out or having a void round produces no return label
            "round_return": profit / wager if wager is not None and status == "complete" else None,

            "round_status": status,
            "round_number": self.steps,
            "true_count": upcoming_count,
            "running_count": 0 if next_betting_view["reshuffle_pending"] else self.counter.running_count,
            "terminal_reason": self.terminal_reason,
        }

        # return the next state, reward, whether the episode ended and round details
        # the training loop will use these values to teach the DQN
        return next_state, reward, self.done, info
