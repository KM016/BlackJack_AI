
import random
from collections import deque

import torch
from torch import nn


# predict one Q-value for each of the six possible betting actions
# the network only sees the six public features provided by our environment
class DQN(nn.Module):

    def __init__(self):
        # initialise PyTorch's neural network class
        super().__init__()

        # use the original two hidden layers of 64 neurons with ReLU activation
        # the first layer takes our six blackjack features as inputs
        # the two hidden layers learn patterns between the inputs and rewards
        # ReLU introduces nonlinearity so the network can learn complex relationships
        # the final layer gives one predicted Q-value for each of our six actions
        self.layers = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, features):
        # return the estimated future reward for each possible wager

        # pass the features through every layer of the neural network
        # each output is a Q-value estimating the total future reward
        # from choosing that action and following the learned policy afterwards
        return self.layers(features)


# remember previous rounds so the DQN can learn from randomly sampled experiences
class ReplayBuffer:

    def __init__(self, capacity=10000):

        # make sure we can store at least one experience
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Replay capacity must be positive")

        # create a memory that stores up to 10000 blackjack experiences
        # when the buffer fills up, the oldest experiences are removed
        # this prevents memory usage from increasing indefinitely
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # store only public features, action masks, actions and observed rewards

        # save what the DQN observed, what it did and what happened afterwards
        # this is called a transition: (state, action, reward, next state, done)
        #
        # we also store the next state's legal action mask
        # because not every betting action will be affordable in every round
        #
        # tuples and float conversions keep the saved experience independent
        # of later changes to the environment's observations
        self.memory.append((
            tuple(state["features"]), action, float(reward),
            tuple(next_state["features"]), bool(done),
            tuple(next_state["legal_mask"]),
        ))

    def sample(self, batch_size, rng):
        # randomly choose previous rounds without changing the saved experiences

        # take a random selection of past experiences from memory
        # sampling without replacement means the same stored experience
        # cannot appear twice within a single training batch
        #
        # random sampling reduces the dependence between consecutive rounds
        # and allows the network to learn from older as well as recent outcomes
        return rng.sample(list(self.memory), batch_size)

    def __len__(self):

        # return how many experiences are currently stored
        # this lets the training loop check whether enough data is available
        return len(self.memory)


# choose only legal actions, either exploring randomly or using the network
# epsilon is the probability of trying a random legal wager during training
def choose_action(network, observation, epsilon=0.0, rng=None):

    # epsilon must be between 0 and 1 because it represents a probability
    # 1 means always explore randomly, while 0 means always use the network
    if not 0 <= epsilon <= 1:
        raise ValueError("Epsilon must be between zero and one")

    # get the legal action mask from the current blackjack observation
    # each of its six values indicates whether a particular wager is allowed
    mask = observation["legal_mask"]

    # reject states with an invalid action mask or no available actions
    if len(mask) != 6 or not any(mask):
        raise ValueError("No legal betting action is available")

    # build a list containing only the indices of legal betting actions
    # for example, (True, True, False, ...) gives choices [0, 1, ...]
    choices = [index for index, allowed in enumerate(mask) if allowed]

    # use the supplied random generator for reproducible training
    # otherwise use Python's default random generator
    rng = rng if rng is not None else random

    # sometimes explore instead of selecting the currently highest Q-value
    # with probability epsilon, choose a random action from the legal choices
    # this helps the DQN discover whether alternative wagers perform better
    if epsilon > 0 and rng.random() < epsilon:
        return rng.choice(choices)

    # illegal actions cannot win, regardless of their predicted Q-values
    # no_grad tells PyTorch that we are making a prediction, not training
    # this avoids tracking gradients when simply choosing a wager
    with torch.no_grad():

        # convert the six numerical features into a PyTorch tensor
        # unsqueeze adds a batch dimension because the network expects batches
        # the resulting input has shape (1, 6): one state with six features
        features = torch.tensor(observation["features"], dtype=torch.float32).unsqueeze(0)

        # predict the Q-value of every possible betting action
        # squeeze removes the extra batch dimension, leaving six values
        scores = network(features).squeeze(0)

        # replace illegal actions' Q-values with negative infinity
        # this ensures an illegal wager can never be selected as the best action
        scores = scores.masked_fill(~torch.tensor(mask, dtype=torch.bool), float("-inf"))

        # select the legal action with the highest predicted Q-value
        # argmax returns the index of that action, which we convert to an integer
        return int(scores.argmax().item())


# calculate DQN targets using only legal actions available in the next state
# terminal rounds have no future reward, so their target is only the observed reward
def calculate_targets(target_network, rewards, next_features, done, next_masks, gamma=1.0):

    # we use the target network to estimate the value of future actions
    # its weights are held fixed between periodic updates during training
    # no_grad prevents this calculation from changing the target network
    with torch.no_grad():

        # predict the Q-values for every possible action in the next states
        next_q = target_network(next_features)

        # exclude actions that would be illegal in those next states
        # negative infinity ensures they cannot be chosen as the best future action
        next_q = next_q.masked_fill(~next_masks, float("-inf"))

        # a terminal state can have an entirely false action mask
        # replace its maximum with zero, since no future action is taken

        # find the highest legal future Q-value for each experience
        # this estimates the best reward we could obtain after the current action
        next_value = next_q.max(dim=1).values

        # if the episode has ended, there is no next action or future reward
        # replace the future Q-value with zero for every terminal state
        next_value = torch.where(done, torch.zeros_like(next_value), next_value)

        # a nonterminal state must have at least one legal future action
        # otherwise its best future value would be invalid
        if not torch.isfinite(next_value).all():
            raise ValueError("A nonterminal next state has no legal action")

        # calculate the Bellman target:
        # target = observed reward + gamma * best future Q-value
        #
        # gamma determines how much future rewards matter
        # our gamma of 1 means we do not discount future bankroll growth
        # for terminal states, the future term is zero
        return rewards + gamma * next_value


# update the online network using a random minibatch of previous rounds
def train_batch(network, target_network, optimiser, batch, gamma=1.0):

    # convert stored experiences into the tensors needed by PyTorch
    # each row of the batch represents one previously observed blackjack round

    # the six features describing each original state
    states = torch.tensor([row[0] for row in batch], dtype=torch.float32)

    # the action index chosen by the DQN during that round
    # long is needed because PyTorch uses integer indices to select Q-values
    actions = torch.tensor([row[1] for row in batch], dtype=torch.long)

    # the actual log-bankroll reward received after playing that round
    rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)

    # the six features describing the state reached after the action
    next_states = torch.tensor([row[3] for row in batch], dtype=torch.float32)

    # whether the blackjack episode ended after each experience
    done = torch.tensor([row[4] for row in batch], dtype=torch.bool)

    # which betting actions were legal in each experience's next state
    next_masks = torch.tensor([row[5] for row in batch], dtype=torch.bool)

    # compare the chosen action's Q-value with the one-step target

    # predict Q-values for all six actions in each stored state
    # gather keeps only the Q-value of the action that was actually taken
    #
    # for example, if we chose action 3, we train its predicted Q-value
    # rather than directly training the other five action predictions
    # squeeze removes the extra dimension after selecting these values
    predictions = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # calculate what those Q-values should move towards
    # each target combines the observed reward with the estimated
    # value of the best legal action in the next state
    targets = calculate_targets(target_network, rewards, next_states, done, next_masks, gamma)

    # use Huber loss to reduce the effect of unusually large errors
    # loss measures the difference between predicted and target Q-values
    # Huber loss behaves like squared error for small differences
    # and is less sensitive to unusually large differences
    loss = nn.functional.smooth_l1_loss(predictions, targets)

    # clear gradients left over from the previous training update
    # otherwise PyTorch would accumulate them across training batches
    optimiser.zero_grad()

    # use backpropagation to calculate how the network weights
    # should change to reduce the prediction error
    loss.backward()

    # prevent excessively large gradients from destabilising learning
    # the total gradient norm is limited to 10
    nn.utils.clip_grad_norm_(network.parameters(), 10.0)

    # use Adam to update the online network's weights
    # the target network is not updated here
    optimiser.step()

    # return the numerical loss so we can track training progress
    return float(loss.item())
