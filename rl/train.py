import argparse
import csv
import math
import random
from pathlib import Path

import torch

from research.study_plan import load_plan
from rl.dqn import DQN, ReplayBuffer, choose_action, train_batch
from rl.environment import BlackjackBettingEnv, FEATURE_NAMES
from rl.policy import save_policy


# use the original DQN training settings for every independent run
# train using 64 randomly sampled previous rounds at a time
BATCH_SIZE = 64
# keep up to 10000 experiences, removing the oldest when the buffer fills
REPLAY_CAPACITY = 10000
# control how quickly Adam changes the network weights
LEARNING_RATE = 0.001
# future log-bankroll rewards have no discount over our finite session
GAMMA = 1.0
# copy the online network into the target network after every 250 updates
TARGET_UPDATE = 250
# reduce random exploration over the first 10000 betting decisions
EPSILON_DECAY = 10000


# gradually reduce random exploration while still trying some new wagers
def exploration_rate(steps):
    # exploration starts at 100% and falls linearly towards 5%
    progress = min(steps, EPSILON_DECAY) / EPSILON_DECAY
    return 1.0 - progress * (1.0 - 0.05)


# train one DQN using the independent episode seeds already saved in our plan
def train(seeds, folder, network_seed, rounds=500, initial_bankroll="1000", unit="1", reward_scale=1.0):
    # use separate, valid random seeds for independent blackjack sessions
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    if type(network_seed) is not int or network_seed < 0:
        raise ValueError("Network seed must be a nonnegative integer")

    # a positive finite multiplier changes learning rewards, not actual blackjack returns
    if type(reward_scale) not in (int, float) or not math.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("Reward scale must be a positive finite number")
    # do not overwrite an earlier trained model or its research results
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Choose a new output directory for this training run")

    # make the network initialization, exploration and replay sampling repeatable
    random.seed(network_seed)
    torch.manual_seed(network_seed)
    torch.set_num_threads(1)
    rng = random.Random(network_seed)

    # the online network learns; the target network supplies fixed reference values
    network = DQN()
    target_network = DQN()
    target_network.load_state_dict(network.state_dict())
    target_network.eval()
    # Adam performs each weight update using randomly sampled previous rounds
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(REPLAY_CAPACITY)
    environment = BlackjackBettingEnv(initial_bankroll, rounds, unit)

    # count betting decisions, optimiser updates and completed episode results
    steps = 0
    updates = 0
    records = []

    # start each episode with a fresh, separately seeded blackjack game
    for episode, seed in enumerate(seeds, start=1):
        # restart the shoe, bankroll and Hi-Lo counter at the start of the session
        state = environment.reset(seed)
        # keep actual log rewards and round statistics separate from scaled replay data
        rewards = []
        losses = []
        bets = []
        sitting_out = 0

        # each action chooses a wager, then Basic Strategy plays the actual hand
        while not environment.done:
            # explore initially, then increasingly use the network's predictions
            action = choose_action(network, state, exploration_rate(steps), rng)
            # the environment returns the unscaled financial reward for this round
            next_state, reward, done, info = environment.step(action)
            # the replay buffer uses scaled rewards to train the DQN
            # keep the original log-bankroll reward for our financial results
            memory.add(state, action, reward * reward_scale, next_state, done)
            rewards.append(reward)
            steps += 1

            # track betting participation and wager sizes for the training log
            if info["wager"] is None:
                sitting_out += 1
            else:
                bets.append(float(info["wager"]))

            # learn from old rounds once we have enough different experiences
            if len(memory) >= BATCH_SIZE:
                # learn from 64 random experiences rather than only the latest round
                batch = memory.sample(BATCH_SIZE, rng)
                losses.append(train_batch(network, target_network, optimiser, batch, GAMMA))
                updates += 1

                # periodically copy the learned Q-values to a stable target network
                # refresh the target network periodically to reduce moving targets
                if updates % TARGET_UPDATE == 0:
                    target_network.load_state_dict(network.state_dict())

            # use the public state from this completed round for the next decision
            state = next_state

        # collect straightforward diagnostics without selecting or tuning a policy
        # the training summary uses original log returns, not scaled rewards
        final_bankroll = float(environment._engine.players["agent"].bankroll)
        records.append({
            "episode": episode, "seed": seed, "rounds": environment.steps,
            "final_bankroll": final_bankroll, "log_growth": math.fsum(rewards),
            "bets": len(bets), "sits": sitting_out,
            "mean_bet": sum(bets) / len(bets) if bets else 0.0,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "updates": updates, "terminal_reason": environment.terminal_reason,
        })
        print(f"Episode {episode}/{len(seeds)}: {environment.steps} rounds, "
              f"{len(bets)} bets, log growth {math.fsum(rewards):+.5f}")

    # save the trained network, exact episode seeds, settings and training log
    # record the new reward multiplier so experiments can be compared later
    settings = {
        "version": "blackjack_dqn_v1", "features": list(FEATURE_NAMES),
        "wagers": [0, 1, 2, 4, 6, 8], "num_decks": 6,
        "initial_bankroll": str(initial_bankroll), "unit": str(unit), "rounds": rounds,
        "network_seed": network_seed, "seeds": list(seeds),
        "architecture": [6, 64, 64, 6], "gamma": GAMMA,
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "replay_capacity": REPLAY_CAPACITY, "target_update": TARGET_UPDATE,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_steps": EPSILON_DECAY,
        "updates": updates, "environment_steps": steps,
        "reward_scale": reward_scale,
    }
    # save the model and settings; this does not retrain or change its predictions
    save_policy(network, settings, folder)
    # write unscaled bankroll results for each episode to a readable CSV
    with (folder / "training_log.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return network, records


# select one of the three separate training seed lists from our research plan
def main():
    # choose the seed group and output directory from the command line
    parser = argparse.ArgumentParser(description="Train one blackjack DQN candidate")
    parser.add_argument("--plan", default="research/study_plan.json")
    parser.add_argument("--run", choices=("train_1", "train_2", "train_3"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--network-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=500)
    # use 1 for the original DQN or 100 for our scaled-reward experiment
    parser.add_argument("--reward-scale", type=float, default=1.0)
    args = parser.parse_args()

    # use only the allocated training seeds, never the validation or evaluation seeds
    # use the allocated training seeds, keeping selection and final seeds untouched
    plan = load_plan(args.plan)
    seeds = plan["seeds"][args.run]
    if not 1 <= args.episodes <= len(seeds):
        raise ValueError("Episodes must fit the allocated training seed list")
    # train with the requested reward multiplier (one means original behaviour)
    train(seeds[:args.episodes], args.output, args.network_seed, args.rounds,
          reward_scale=args.reward_scale)


if __name__ == "__main__":
    main()
