import csv
import hashlib
import json
import random
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from strategies.betting import decimal_value


# calculate empirical return distributions separately for each true-count range

def parse_bins(values):
    # bin boundaries must be sorted and must not repeat
    if not isinstance(values, list) or not values:
        raise ValueError("Provide a nonempty list of bin boundaries")
    bins = [decimal_value(value, "bin") for value in values]
    if bins != sorted(set(bins)):
        raise ValueError("Bin boundaries must be increasing and unique")
    return bins


def get_bin(true_count, boundaries):
    # boundaries belong to the interval on their right
    return bisect_right(boundaries, decimal_value(true_count, "true count"))


def get_label(index, boundaries):
    # name the true-count range using its lower and upper boundaries
    left = str(boundaries[index - 1]) if index else "-inf"
    right = str(boundaries[index]) if index < len(boundaries) else "inf"
    return f"[{left}, {right})"


def sample_statistics(values):
    # use sample variance, dividing by n - 1 rather than n
    if not values:
        return None, None
    mean = sum(values, Decimal(0)) / len(values)
    variance = None
    if len(values) > 1:
        variance = sum(((x - mean) ** 2 for x in values), Decimal(0)) / (len(values) - 1)
    return mean, variance


def percentile(values, probability):
    # interpolate between sorted estimates for a percentile
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def bootstrap_statistics(session_values, replicates, seed):
    # summarise each independent session once so large studies run efficiently
    session_summaries = []
    for values in session_values.values():
        session_summaries.append((len(values), sum(values, Decimal(0)),
                                  sum((x * x for x in values), Decimal(0))))

    # use a fixed random seed so the bootstrap results can be repeated
    rng = random.Random(seed)
    means = []
    variances = []
    # repeatedly sample complete sessions and recalculate the statistics
    for _ in range(replicates):
        count = 0
        total = Decimal(0)
        total_squared = Decimal(0)
        for _ in session_summaries:
            n, s, ss = session_summaries[rng.randrange(len(session_summaries))]
            count += n
            total += s
            total_squared += ss
        if count:
            means.append(total / count)
        if count > 1:
            variance = (total_squared - total * total / count) / (count - 1)
            variances.append(variance)

    # calculate the 95% confidence interval from the bootstrap estimates
    def interval(estimates):
        if len(estimates) < 2:
            return None
        return [percentile(estimates, Decimal("0.025")),
                percentile(estimates, Decimal("0.975"))]

    return interval(means), interval(variances), len(means), len(variances)


def bootstrap_mean_ci(session_values, replicates, seed):
    # expose the mean interval for straightforward standalone testing
    mean_ci, _, mean_count, _ = bootstrap_statistics(session_values, replicates, seed)
    return mean_ci, mean_count


def read_samples(folder, role, plan):
    # load only data collected using the approved plan and requested seed role
    from research.study_plan import plan_hash, source_hashes

    # find the saved dataset and its experiment settings
    folder = Path(folder)
    metadata = json.loads((folder / f"{role}_metadata.json").read_text())
    path = folder / f"{role}_samples.csv"

    # check that the dataset finished and belongs to the requested stage
    if metadata["status"] != "complete" or metadata["role"] != role:
        raise ValueError("Dataset is incomplete or has the wrong role")
    # check that the dataset uses the expected plan and random seeds
    if metadata["plan_hash"] != plan_hash(plan) or metadata["seeds"] != plan["seeds"][role]:
        raise ValueError("Dataset does not match the approved seed plan")
    # check that the saved data file has not changed
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Dataset hash mismatch")
    # check that the simulation source files match the recorded versions
    if metadata["code_hashes"] != source_hashes():
        raise ValueError("Simulation code has changed since data generation")

    # prepare to read each round and identify duplicates or missing results
    rows = []
    seen = set()
    statuses = Counter()
    with path.open(newline="") as stream:
        # read the recorded blackjack rounds one at a time
        for raw in csv.DictReader(stream):
            seed = int(raw["seed"])
            round_number = int(raw["round"])
            # identify a round using its session seed and round number
            key = (seed, round_number)
            if (seed not in metadata["seeds"] or key in seen
                    or not 1 <= round_number <= plan["calibration"]["rounds_per_session"]
                    or raw["session"] != f"{role}:{seed}"):
                raise ValueError("Unexpected, duplicate or missing session record")
            seen.add(key)
            # reject incomplete rounds and record the outcome status
            if raw["status"] not in ("complete", "void"):
                raise ValueError("Unsuccessful rounds cannot enter calibration")
            statuses[raw["status"]] += 1
            # check that the shoe and action indicators are valid
            if not raw["shoe"].startswith(f"{role}:{seed}:"):
                raise ValueError("Incorrect shoe identifier")
            if any(raw[field] not in ("0", "1") for field in ("split", "double", "natural", "all_hands_push")):
                raise ValueError("Invalid hand outcome indicator")
            if raw["reshuffle_pending"] not in ("0", "1"):
                raise ValueError("Invalid reshuffle indicator")
            # convert the numerical CSV values back into Decimal amounts
            row = dict(raw)
            row["true_count"] = decimal_value(raw["true_count"], "true count")
            row["initial_wager"] = decimal_value(raw["initial_wager"], "initial wager")
            row["net_profit"] = decimal_value(raw["net_profit"], "net profit")
            if row["initial_wager"] != 1:
                raise ValueError("Calibration must use one-unit fixed stakes")
            # independently check the true count from cards already revealed
            running_count = int(raw["running_count"])
            cards_seen = int(raw["cards_seen"])
            remaining = 312 - cards_seen
            if not 0 < remaining <= 312 or abs(running_count) > cards_seen:
                raise ValueError("Invalid public card count")
            expected_tc = Decimal(running_count * 52) / Decimal(remaining)
            if expected_tc != row["true_count"]:
                raise ValueError("True count does not match the public card count")
            if raw["reshuffle_pending"] == "1" and (running_count != 0 or cards_seen != 0):
                raise ValueError("Pending reshuffle must reset the betting count")
            # keep voided rounds separate from ordinary profit and loss results
            if raw["status"] == "void":
                if raw["return"] or row["net_profit"] != 0:
                    raise ValueError("Voided rounds must have zero profit and no return label")
                row["return"] = None
            else:
                row["return"] = decimal_value(raw["return"], "return")
                if row["return"] != row["net_profit"] / row["initial_wager"] or not -8 <= row["return"] <= 8:
                    raise ValueError("Incorrect round return label")
            rows.append(row)

    # make sure every planned session contains every expected round
    expected = {(seed, number) for seed in metadata["seeds"]
                for number in range(1, plan["calibration"]["rounds_per_session"] + 1)}
    if seen != expected or len(rows) != metadata["attempts"]:
        raise ValueError("Missing attempted rounds")
    if statuses["complete"] != metadata["complete"] or statuses["void"] != metadata["void"]:
        raise ValueError("Round status counts do not reconcile")
    return metadata, rows


def bootstrap_all_bins(groups, seeds, replicates, seed):
    # resample the same full sessions across every bin in each bootstrap draw
    # this matches the original joint whole-session bootstrap method
    # reduce each session to its return count, sum and squared sum for every bin
    bin_count = len(groups)
    summaries = []
    for group in groups:
        by_session = []
        for session in seeds:
            values = group[session]
            by_session.append((len(values), sum(values, Decimal(0)),
                               sum((value * value for value in values), Decimal(0))))
        summaries.append(by_session)

    # prepare reproducible bootstrap results for all true-count bins
    rng = random.Random(seed)
    mean_draws = [[] for _ in groups]
    variance_draws = [[] for _ in groups]
    # choose whole sessions once per draw, then use them for every bin
    for _ in range(replicates):
        selected = [rng.randrange(len(seeds)) for _ in seeds]
        for index in range(bin_count):
            count = sum(summaries[index][choice][0] for choice in selected)
            total = sum((summaries[index][choice][1] for choice in selected), Decimal(0))
            squared = sum((summaries[index][choice][2] for choice in selected), Decimal(0))
            if count:
                mean_draws[index].append(total / count)
            if count > 1:
                variance_draws[index].append((squared - total * total / count) / (count - 1))
    return mean_draws, variance_draws


def bootstrap_uncertainty(values):
    # report the percentile interval and standard error of the bootstrap estimates
    if len(values) < 2:
        return None, None
    mean = sum(values, Decimal(0)) / len(values)
    variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / (len(values) - 1)
    return [percentile(values, Decimal("0.025")),
            percentile(values, Decimal("0.975"))], variance.sqrt()


def summarise(rows, seeds, role, settings):
    # calculate original empirical moments and distributions by pre-round true count
    boundaries = parse_bins(settings["bins"])
    # keep returns separate by true-count bin and simulation session
    groups = [{seed: [] for seed in seeds} for _ in range(len(boundaries) + 1)]
    shoes = [set() for _ in groups]
    counters = [Counter() for _ in groups]
    voids = 0
    # group completed round returns and count special blackjack outcomes
    for row in rows:
        if row["status"] == "void":
            voids += 1
            continue
        index = get_bin(row["true_count"], boundaries)
        groups[index][int(row["seed"])].append(row["return"])
        if "shoe" in row:
            shoes[index].add(row["shoe"])
        counters[index]["positive_return" if row["return"] > 0 else
                        "negative_return" if row["return"] < 0 else "zero_return"] += 1
        for field, label in (("double", "had_double"), ("split", "had_split"),
                             ("natural", "had_natural"), ("all_hands_push", "all_hands_push")):
            if field in row:
                counters[index][label] += int(row[field])

    # all bins use the same session-selection draw, as in the original research
    mean_draws, variance_draws = bootstrap_all_bins(
        groups, seeds, settings["bootstrap_replicates"], settings["bootstrap_seed"]
    )
    # calculate the estimated return and uncertainty in each true-count bin
    bins = []
    for index, group in enumerate(groups):
        values = [value for session in group.values() for value in session]
        mean, variance = sample_statistics(values)
        ci, mean_se = bootstrap_uncertainty(mean_draws[index])
        variance_ci, variance_se = bootstrap_uncertainty(variance_draws[index])
        bootstrap_count = len(mean_draws[index])
        variance_bootstrap_count = len(variance_draws[index])
        contributing_sessions = sum(bool(session) for session in group.values())
        # count how often each possible round return occurred
        distribution = Counter(str(value) for value in values)

        # follow the original fixed support and uncertainty checks
        reasons = []
        if len(values) < settings["min_samples"]:
            reasons.append("insufficient_samples")
        if contributing_sessions < settings["min_sessions"]:
            reasons.append("insufficient_sessions")
        if variance is None or variance <= Decimal(settings["min_variance"]):
            reasons.append("invalid_variance")
        if (ci is None or variance_ci is None
                or min(bootstrap_count, variance_bootstrap_count)
                < Decimal(settings["bootstrap_replicates"]) * Decimal("0.95")):
            reasons.append("insufficient_bootstrap_coverage")
        if ci is not None and ci[1] - ci[0] > Decimal(settings["max_ci_width"]):
            reasons.append("wide_confidence_interval")
        # only use a bin if it passes every support check
        supported = not reasons

        # save the bin statistics, confidence intervals and return frequencies
        bins.append({
            "label": get_label(index, boundaries),
            "count": len(values), "sessions": contributing_sessions,
            "shoes": len(shoes[index]),
            "mean": str(mean) if mean is not None else None,
            "variance": str(variance) if variance is not None else None,
            "mean_ci": [str(x) for x in ci] if ci else None,
            "variance_ci": [str(x) for x in variance_ci] if variance_ci else None,
            "mean_standard_error": str(mean_se) if mean_se is not None else None,
            "variance_standard_error": str(variance_se) if variance_se is not None else None,
            "bootstrap_coverage": bootstrap_count,
            "variance_bootstrap_coverage": variance_bootstrap_count,
            "supported": supported,
            "positive_edge_supported": supported and ci[0] > 0,
            "unsupported_reasons": reasons,
            "frequencies": {name: str(Decimal(counters[index][name]) / len(values))
                            if values else None for name in
                            ("positive_return", "negative_return", "zero_return", "had_double",
                             "had_split", "had_natural", "all_hands_push")},
            "distribution": [{"return": value, "count": count} for value, count in
                             sorted(distribution.items(), key=lambda item: Decimal(item[0]))],
            # preserve per-session counts for auditing without reading the raw CSV
            "session_return_counts": {
                str(seed): dict(sorted(Counter(str(x) for x in session).items()))
                for seed, session in group.items()
            }
        })

    return {"bins": bins, "complete_rounds": sum(item["count"] for item in bins),
            "void_rounds": voids, "sessions": len(seeds), "role": role}


# store a fitted model and return predictions for a public true count
class CalibrationModel:
    def __init__(self, document):
        # store a separate copy of the fitted calibration results
        self._document = deepcopy(document)
        if document.get("version") != "calibration_v1" or document.get("role") != "calibration":
            raise ValueError("Model must have been fitted on calibration data")
        # check that every true-count bin is present and correctly labelled
        self.boundaries = parse_bins(document["settings"]["bins"])
        if len(document["summary"]["bins"]) != len(self.boundaries) + 1:
            raise ValueError("Incorrect number of calibrated bins")
        for index, item in enumerate(document["summary"]["bins"]):
            if item["label"] != get_label(index, self.boundaries):
                raise ValueError("Incorrect true-count bin label")
            # verify the statistics and evidence thresholds of supported bins
            if item["supported"]:
                mean = decimal_value(item["mean"], "mean")
                variance = decimal_value(item["variance"], "variance")
                ci = [decimal_value(x, "CI") for x in item["mean_ci"]]
                if variance <= Decimal(document["settings"]["min_variance"]) or ci[0] > ci[1] or not -8 <= mean <= 8:
                    raise ValueError("Invalid supported estimate")
                settings = document["settings"]
                if (item["count"] < settings["min_samples"]
                        or item["sessions"] < settings["min_sessions"]
                        or ci[1] - ci[0] > Decimal(settings["max_ci_width"])
                        or item["bootstrap_coverage"] < Decimal(settings["bootstrap_replicates"]) * Decimal("0.95")
                        or item["variance_bootstrap_coverage"]
                        < Decimal(settings["bootstrap_replicates"]) * Decimal("0.95")):
                    raise ValueError("Supported bin fails the original evidence thresholds")

    def predict(self, true_count):
        # use the fitted bin without inspecting new evaluation outcomes
        item = self._document["summary"]["bins"][get_bin(true_count, self.boundaries)]
        # do not make predictions for bins without enough reliable evidence
        if not item["supported"]:
            return None
        return {
            "bin": item["label"],
            "mean": Decimal(item["mean"]),
            "variance": Decimal(item["variance"]),
            "mean_ci": [Decimal(value) for value in item["mean_ci"]],
            "distribution": deepcopy(item["distribution"])
        }

    def save(self, path):
        # do not overwrite an existing model accidentally
        with Path(path).open("x") as stream:
            json.dump(self._document, stream, indent=2)
            stream.write("\n")

    @classmethod
    def load(cls, path):
        # recreate a calibration model using its saved JSON results
        return cls(json.loads(Path(path).read_text()))

    def get_document(self):
        # return a copy so callers cannot change the fitted model
        return deepcopy(self._document)


def fit_model(folder, plan, output):
    # fit only calibration samples; independent validation data is never used here
    from research.study_plan import plan_hash

    from research.study_plan import validate_plan
    validate_plan(plan)
    # load only the completed calibration dataset
    metadata, rows = read_samples(folder, "calibration", plan)
    # estimate the returns, variance and uncertainty for each count range
    summary = summarise(rows, plan["seeds"]["calibration"], "calibration", plan["calibration"])
    document = {"version": "calibration_v1", "role": "calibration",
                "plan_hash": plan_hash(plan), "population": metadata["population"],
                "settings": plan["calibration"], "metadata": metadata, "summary": summary}
    # validate the fitted estimates and save the finished model
    model = CalibrationModel(document)
    model.save(output)
    return model
