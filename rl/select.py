"""Select one frozen DQN using the reserved selection seeds only."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

from research.study_plan import load_plan, plan_hash
from rl.dqn import choose_action
from rl.environment import BlackjackBettingEnv
from rl.policy import load_policy


# compare exactly six unscaled models in this fixed order
# reward-scaled models are recorded separately, not eligible for selection
CANDIDATE_NAMES = (
    "original_1", "original_2", "original_3",
    "extended_1", "extended_2", "extended_3",
)


# identify a saved file without changing or retraining its contents
def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# find the three original and three longer-trained model folders
def candidate_folders(original_root, extended_root):
    original_root = Path(original_root)
    extended_root = Path(extended_root)
    return {
        **{f"original_{i}": original_root / f"train_{i}" for i in range(1, 4)},
        **{f"extended_{i}": extended_root / f"train_{i}" for i in range(1, 4)},
    }


# verify that the candidate belongs to our original or extended research plan
# this also prevents accidentally mixing reward-scaled or other trained models
def validate_candidate(settings, plan, name):
    run_number = int(name.rsplit("_", 1)[1])
    expected_episodes = 100 if name.startswith("original_") else 500
    expected_seeds = plan["seeds"][f"train_{run_number}"][:expected_episodes]

    if settings.get("seeds") != expected_seeds:
        raise ValueError(f"{name}: training seeds do not match the research plan")
    if settings.get("network_seed") != 101 * run_number:
        # the second and third model initialization seeds are 202 and 303
        raise ValueError(f"{name}: unexpected network initialization seed")
    if float(settings.get("reward_scale", 1.0)) != 1.0:
        raise ValueError(f"{name}: reward-scaled models are excluded")
    if (settings.get("initial_bankroll") != "1000" or settings.get("unit") != "1"
            or settings.get("rounds") != 500 or settings.get("epsilon_decay_steps") != 10000):
        raise ValueError(f"{name}: different bankroll, unit, horizon or exploration schedule")
    if settings.get("environment_steps") != expected_episodes * 500:
        raise ValueError(f"{name}: incomplete recorded training steps")


# play a complete independent session without exploration or optimiser updates
# only public information is passed to the saved neural network
def evaluate_session(network, seed, name):
    environment = BlackjackBettingEnv(initial_bankroll="1000", rounds=500, unit="1")
    state = environment.reset(seed)
    start = float(environment.initial_bankroll)
    peak = start
    largest_drawdown = 0.0
    rewards = []
    bets = 0
    total_wager = 0.0
    void_rounds = 0

    while not environment.done:
        action = choose_action(network, state, epsilon=0.0)
        state, reward, done, info = environment.step(action)
        rewards.append(reward)
        bankroll = float(info["bankroll_after"])
        peak = max(peak, bankroll)
        largest_drawdown = max(largest_drawdown, 1.0 - bankroll / peak)
        if info["wager"] is not None:
            bets += 1
            total_wager += float(info["wager"])
        if info["round_status"] == "void":
            void_rounds += 1

    final = float(environment._engine.players["agent"].bankroll)
    log_growth = math.fsum(rewards)
    if not math.isclose(log_growth, math.log(final / start), rel_tol=1e-10, abs_tol=1e-12):
        raise ArithmeticError(f"{name} seed {seed}: log rewards disagree with bankroll")
    if environment.steps != 500 or environment.terminal_reason != "horizon":
        raise RuntimeError(f"{name} seed {seed}: session ended before the planned horizon")

    return {
        "candidate": name, "seed": seed, "rounds": environment.steps,
        "final_bankroll": final, "log_growth": log_growth,
        "net_profit": final - start, "max_drawdown": largest_drawdown,
        "bets": bets, "sits": environment.steps - bets,
        "mean_bet": total_wager / bets if bets else 0.0,
        "void_rounds": void_rounds,
    }


# summarize separate 500-round sessions, not individual correlated rounds
def summarize_candidate(rows, name):
    if not rows:
        raise ValueError("Cannot summarize an empty set of sessions")
    return {
        "candidate": name,
        "sessions": len(rows),
        "mean_log_growth": statistics.mean(row["log_growth"] for row in rows),
        "mean_final_bankroll": statistics.mean(row["final_bankroll"] for row in rows),
        "mean_net_profit": statistics.mean(row["net_profit"] for row in rows),
        "mean_max_drawdown": statistics.mean(row["max_drawdown"] for row in rows),
        "betting_rate": sum(row["bets"] for row in rows) / sum(row["rounds"] for row in rows),
    }


# primary selection criterion is largest mean session log growth
# ties within 1e-12 use smaller mean drawdown, then fixed candidate order
def choose_winner(summaries):
    if not summaries:
        raise ValueError("No DQN candidates to choose from")
    winner = summaries[0]
    for candidate in summaries[1:]:
        difference = candidate["mean_log_growth"] - winner["mean_log_growth"]
        if difference > 1e-12:
            winner = candidate
        elif abs(difference) <= 1e-12:
            if candidate["mean_max_drawdown"] < winner["mean_max_drawdown"] - 1e-12:
                winner = candidate
            elif abs(candidate["mean_max_drawdown"] - winner["mean_max_drawdown"]) <= 1e-12:
                if CANDIDATE_NAMES.index(candidate["candidate"]) < CANDIDATE_NAMES.index(winner["candidate"]):
                    winner = candidate
    return winner


# write records without overwriting a previous formal selection experiment
def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# evaluate the fixed candidate set once on the same reserved selection seeds
def select(plan_path, original_root, extended_root, output):
    plan_path = Path(plan_path)
    plan = load_plan(plan_path)
    selection_seeds = plan["seeds"]["selection"]
    if len(selection_seeds) != 30:
        raise ValueError("Formal model selection requires the 30 reserved selection seeds")

    folders = candidate_folders(original_root, extended_root)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Selection output already exists; do not overwrite it")

    # validate all six candidates before running any reserved selection session
    torch.set_num_threads(1)
    models = {}
    model_hashes = {}
    for name in CANDIDATE_NAMES:
        folder = folders[name]
        policy = load_policy(folder)
        validate_candidate(policy.settings, plan, name)
        models[name] = policy.network.eval()
        model_hashes[name] = {
            "folder": str(folder),
            "policy_sha256": file_hash(folder / "policy.pt"),
            "settings_sha256": file_hash(folder / "settings.json"),
        }

    output.mkdir(parents=True)
    all_rows = []
    summaries = []
    # every model sees the same episode seeds; policies may consume different cards
    for name in CANDIDATE_NAMES:
        rows = []
        for index, seed in enumerate(selection_seeds, start=1):
            row = evaluate_session(models[name], seed, name)
            rows.append(row)
            all_rows.append(row)
            if index % 10 == 0:
                print(f"{name}: {index}/{len(selection_seeds)} sessions completed", flush=True)
        summary = summarize_candidate(rows, name)
        summaries.append(summary)
        print(f"{name}: mean log growth {summary['mean_log_growth']:+.6f}", flush=True)

    winner = choose_winner(summaries)
    write_csv(output / "sessions.csv", all_rows)
    write_csv(output / "summary.csv", summaries)
    # save the selected existing file path, rather than copying or rewriting weights
    decision = {
        "selected_candidate": winner["candidate"],
        "selected_policy_folder": str(folders[winner["candidate"]]),
        "selection_method": "highest mean session log growth; ties within 1e-12: smaller mean maximum drawdown, then candidate order",
        "selection_seeds": selection_seeds,
        "research_plan_sha256": plan_hash(plan),
        "research_plan_file_sha256": file_hash(plan_path),
        "candidates": list(CANDIDATE_NAMES),
        "model_files": model_hashes,
        "winner_summary": winner,
        "final_evaluation_seeds_used": False,
    }
    (output / "selected.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(f"Selected {winner['candidate']} -> {folders[winner['candidate']]}", flush=True)
    return decision


def main():
    parser = argparse.ArgumentParser(description="Select one frozen DQN from six unscaled candidates")
    parser.add_argument("--plan", default="research/study_plan_dqn_extended.json")
    parser.add_argument("--original-root", default="outputs/dqn")
    parser.add_argument("--extended-root", default="outputs/dqn_extended")
    parser.add_argument("--output", default="outputs/dqn_selection")
    args = parser.parse_args()
    select(args.plan, args.original_root, args.extended_root, args.output)


if __name__ == "__main__":
    main()
