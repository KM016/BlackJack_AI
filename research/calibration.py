import argparse
import csv
import hashlib
import json
from pathlib import Path

from blackjack.engine import BlackjackEngine
from blackjack.player import Player
from strategies.basic import BasicStrategy
from strategies.flat import FlatBetting
from strategies.hi_lo import HiLoCounter
from strategies.runner import run_round
from research.study_plan import load_plan, plan_hash, validate_plan, source_hashes
from research.calibration_model import fit_model


# collect unbiased one unit blackjack returns before fitting a betting model
FIELDS = ("session", "seed", "round", "shoe", "reshuffle_pending", "true_count",
          "running_count", "cards_seen", "status", "initial_wager", "net_profit",
          "return", "hand_count", "split", "double", "natural", "all_hands_push")
POPULATION = "two_seats_fixed_one_unit_s17_das"


def describe_round(record):
    # calculate the total return from all of the measured players hands
    player = record["players"]["agent"]
    wager = player["initial_wager"]
    hands = player["hands"]
    profit = sum((hand["net_profit"] for hand in hands), start=wager * 0)

    # make sure the hand profits match the recorded profit and bankroll change
    if not hands or profit != player["net_profit"]:
        raise ValueError("Hand profits do not reconcile with the player's bankroll")
    if profit != player["bankroll_end"] - player["bankroll_start"]:
        raise ValueError("Round return does not reconcile with bankroll change")

     # check that every initial wager is one unit
    if wager != 1:
        raise ValueError("Calibration requires a fixed one unit wager")

    # cancelled rounds must return all money to the player
    if record["state"] == "void" and profit != 0:
        raise ValueError("A voided round must be fully refunded")

    # record the rounds return and whether any special actions occurred
    return {
        "initial_wager": wager,
        "net_profit": profit,
        "return": profit / wager if record["state"] == "complete" else "",
        "hand_count": len(hands),
        "split": int(any(hand["from_split"] for hand in hands)),
        "double": int(any(hand["doubled"] for hand in hands)),
        "all_hands_push": int(all(hand["outcome"] == "push" for hand in hands)),
        "natural": int(any(
            not hand["from_split"] and len(hand["cards"]) == 2
            and "Ace" in [card["value"] for card in hand["cards"]]
            and any(card["value"] in ("10", "Jack", "Queen", "King") for card in hand["cards"])
            for hand in hands
        ))
    }


def collect_session(seed, role, rounds):
    # the first seat always wagers to expose cards, the second is measured
    seats = ("reference", "agent")

    # create a blackjack game with large bankrolls to avoid funding restrictions
    engine = BlackjackEngine(
        {seat: Player("1000000000") for seat in seats}, seed=seed, retain_history=False
    )

    # both players use Basic Strategy and always bet one unit
    playing = {seat: BasicStrategy() for seat in seats}
    betting = {seat: FlatBetting("1") for seat in seats}

    # track the publicly revealed cards and which shoe we are using
    counter = HiLoCounter(num_decks=6)
    shoe_number = 1

    # play the required number of rounds for this session
    for round_number in range(1, rounds + 1):
        # record only information available before this round starts
        observation = engine.betting_observation("agent")
        true_count = counter.get_betting_true_count(observation)
        pending = observation["reshuffle_pending"]

        # update the shoe number whenever the cards are reshuffled
        if pending:
            shoe_number += 1

        # a one unit initial bet may require up to eight units after splits/doubles
        for seat in seats:
            if engine.players[seat].bankroll < 8:
                raise ValueError("Calibration bankroll cannot fund all legal actions")

        # play the round automatically and check that it finished
        result = run_round(engine, playing, betting)
        if result["status"] not in ("complete", "void"):
            raise ValueError("Calibration round did not finish")

        # get the completed rounds results from the engine
        record = engine.history.last_round
        if record["round_number"] != round_number:
            raise ValueError("Recorded round number does not match")

        # save the true count and other information recorded before dealing
        row = {
            "session": f"{role}:{seed}", "seed": seed, "round": round_number,
            "shoe": f"{role}:{seed}:{shoe_number}",
            "reshuffle_pending": int(pending), "true_count": true_count,
            "running_count": 0 if pending else counter.running_count,
            "cards_seen": 0 if pending else counter.cards_seen,
            "status": result["status"]
        }
        # add the rounds profit, return and playing information
        row.update(describe_round(record))
        yield row   # return this rounds data to whoever is collecting the results


def generate_samples(folder, role, plan):
    # create calibration and validation datasets using their own reserved seeds
    if role not in ("calibration", "calibration_validation"):
        raise ValueError("Only calibration or calibration_validation may generate samples")
    validate_plan(plan)

    # create the output folder and paths for our results
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{role}_samples.csv"
    metadata_path = folder / f"{role}_metadata.json"

    # prevent existing research results from being overwritten
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError("Choose a new output directory; do not overwrite prior samples")

    # get the random seeds and number of rounds from our research plan
    seeds = plan["seeds"][role]
    rounds = plan["calibration"]["rounds_per_session"]

    # record the settings used to generate this dataset
    metadata = {
        "version": "calibration_data_v1", "role": role, "population": POPULATION,
        "plan_hash": plan_hash(plan), "code_hashes": source_hashes(),
        "status": "running", "seeds": seeds,
        "rounds_per_session": rounds, "attempts": 0, "complete": 0, "void": 0,
        "starting_bankroll": "1000000000", "stake": "1",
        "rules": {"decks": 6, "penetration": "0.75", "dealer": "S17",
                  "split": "same rank, four hands, no resplit Aces", "DAS": True,
                  "blackjack": "3:2", "surrender": False, "insurance": False},
        "count": "public Hi-Lo before each round; pending reshuffle resets to 0"
    }

    # save the experiment settings before generating any rounds
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    try:
        # create a CSV file to store each simulated round
        with csv_path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()

            # generate all rounds for every seed in this dataset
            for seed in seeds:
                for row in collect_session(seed, role, rounds):

                    # write the round's results and update our counts
                    writer.writerow(row)
                    metadata["attempts"] += 1
                    metadata[row["status"]] += 1

    # mark the dataset as failed if an error occurs
    except Exception:
        metadata["status"] = "failed"
        raise

    # otherwise mark the dataset as complete
    else:
        metadata["status"] = "complete"

    # save the final metadata even if generation failed
    finally:
        if csv_path.exists():
            metadata["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    # return the dataset information
    return metadata


def main():
    # run the reproducible research workflow from the terminal
    parser = argparse.ArgumentParser(description="Generate or fit blackjack calibration data")
    commands = parser.add_subparsers(dest="command", required=True)

    # define the arguments needed to generate simulation data
    generate = commands.add_parser("generate")
    generate.add_argument("--plan", required=True)
    generate.add_argument("--role", choices=("calibration", "calibration_validation"), required=True)
    generate.add_argument("--output", required=True)

    # define the arguments needed to fit the calibration model
    fit = commands.add_parser("fit")
    fit.add_argument("--plan", required=True)
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)

    # read the users command and load the research settings
    args = parser.parse_args()
    plan = load_plan(args.plan)

    # generate data or fit the model depending on the chosen command
    if args.command == "generate":
        metadata = generate_samples(args.output, args.role, plan)
        print(f"Saved {metadata['attempts']} {args.role} attempts")
    else:
        model = fit_model(args.data, plan, args.output)
        print(f"Saved calibration model with {len(model.get_document()['summary']['bins'])} true count bins")


if __name__ == "__main__":
    main()
