
import json
from decimal import Decimal
from pathlib import Path

import torch

from rl.dqn import DQN, choose_action
from rl.environment import FEATURE_NAMES, public_state
from strategies.hi_lo import HiLoCounter
from strategies.wager_grid import wager_for_action


# save a trained network and the settings needed to use it later
def save_policy(network, settings, folder):

    # create the output folder if it does not already exist
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # save the neural network weights and training settings separately
    # policy.pt stores what the network learned during training
    # settings.json stores the information needed to use it correctly
    weights_path = folder / "policy.pt"
    settings_path = folder / "settings.json"

    # never replace a model that has already been saved
    # this prevents accidentally losing the results of an earlier experiment
    if weights_path.exists() or settings_path.exists():
        raise FileExistsError("Choose a new folder instead of overwriting a policy")

    # save the trained network's weights using PyTorch
    # we only need its learned parameters, not the entire training process
    torch.save(network.state_dict(), weights_path)

    # save the training settings as a readable JSON file
    # these include the network features, betting actions and game settings
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


# make greedy betting decisions using a saved DQN, without training it again
class DQNBetting:

    def __init__(self, network, settings):

        # training and inference must have the same features and action grid

        # check that the saved model uses the format expected by our code
        if settings.get("version") != "blackjack_dqn_v1":
            raise ValueError("Unknown DQN policy format")

        # the network must receive the same six inputs, in the same order,
        # that it received when we originally trained it
        if settings.get("features") != list(FEATURE_NAMES):
            raise ValueError("DQN features do not match the training model")

        # the six possible betting actions must also match the training setup
        # otherwise the network's predictions could correspond to the wrong bets
        if settings.get("wagers") != [0, 1, 2, 4, 6, 8]:
            raise ValueError("DQN wager choices do not match training")

        # check that the saved model uses our six-deck game
        # and was trained with a valid number of rounds per session
        if settings.get("num_decks") != 6 or settings.get("rounds", 0) < 1:
            raise ValueError("Invalid saved DQN rules or horizon")

        # switch the network into evaluation mode
        # we are now using its learned predictions rather than training it
        self.network = network.eval()

        # keep the original settings so we can recreate the training inputs
        self.settings = settings

        # restore the starting bankroll and unit size used during training
        # Decimal keeps monetary values consistent with the blackjack engine
        self.initial_bankroll = Decimal(str(settings["initial_bankroll"]))
        self.unit = Decimal(str(settings["unit"]))

        # initialise the card counter and bankroll tracking for a new session
        self.reset()

    # clear the card counter and highest bankroll before a new independent session
    def reset(self):

        # begin with a fresh Hi-Lo count so information from previous
        # blackjack sessions cannot influence the next one
        self.counter = HiLoCounter(num_decks=6)

        # reset the highest recorded bankroll to the starting amount
        # this is needed to calculate drawdown, one of our six DQN inputs
        self.peak_bankroll = self.initial_bankroll

        # clear the previous betting decision
        self.last_decision = None

    # use the trained DQN to choose a legal wager from the current game state
    def choose_wager(self, observation):

        # use exactly the same public feature calculation as DQN training

        # make sure the current game has the same number of decks
        # that our network was trained to understand
        if observation["num_decks"] != 6:
            raise ValueError("Saved DQN requires six decks")

        # get the current bankroll from the public betting observation
        bankroll = observation["bankroll"]

        # update the highest bankroll reached so far
        # the difference between this peak and our current bankroll
        # helps the network understand how much money we have lost
        self.peak_bankroll = max(self.peak_bankroll, bankroll)

        # convert the current game into the same six numerical features
        # used during training: true count, running count, remaining cards,
        # relative bankroll, drawdown and remaining rounds
        #
        # this also calculates which wagers are currently legal
        # so the network cannot select an unaffordable bet
        state = public_state(
            observation, self.counter, self.initial_bankroll,
            self.peak_bankroll, observation["round_number"],
            self.settings["rounds"], self.unit,
        )

        # evaluation is always greedy, with no exploration or parameter updates
        # epsilon = 0 means the network always selects its highest-valued
        # legal action rather than occasionally choosing a random wager
        action = choose_action(self.network, state, epsilon=0.0)

        # convert the network's selected action into an actual betting amount
        # action 0 returns None, meaning the agent chooses to sit out
        wager = wager_for_action(action, self.unit)

        # store the selected action and wager for later analysis
        # this helps us understand what the trained DQN decided to do
        self.last_decision = {"action_index": action, "wager": wager}

        # return the wager so the blackjack engine can play the next round
        return wager


# rebuild the betting policy from plain settings and saved model weights
# loading never retrains the network or accesses any blackjack outcomes
def load_policy(folder):

    # locate the folder containing our saved network and settings
    folder = Path(folder)

    # read the original training settings from the JSON file
    # we need these to interpret the network's inputs and outputs correctly
    settings = json.loads((folder / "settings.json").read_text())

    # create a fresh DQN with the same architecture used during training
    # at this point its weights have not yet been restored
    network = DQN()

    # load the learned weights into the new network
    # map_location="cpu" allows us to load the model on a CPU
    # weights_only=True loads the stored weights without loading arbitrary
    # Python objects from the file
    network.load_state_dict(torch.load(folder / "policy.pt", map_location="cpu", weights_only=True))

    # create a betting strategy using the restored network
    # the resulting policy is ready to make decisions without further training
    return DQNBetting(network, settings)
