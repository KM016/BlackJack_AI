"""Validate saved final evidence and analyse independent, paired sessions.

This module never simulates a hand, trains a model, or changes historical data.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


STRATEGIES = (
    "flat", "hi_lo", "calibrated_kelly", "model_based_log",
    "dqn", "dqn_runner_up", "never_bet",
)
INITIAL_BANKROLL = 1000
ROUNDS = 500
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 25002
SESSION_FIELDS = (
    "strategy", "seed", "rounds", "final_bankroll", "net_profit", "log_growth",
    "max_drawdown", "return_std", "bets", "sits", "mean_bet", "void_rounds",
    "absorbed_rounds", "terminal_reason",
)
INTEGER_FIELDS = ("seed", "rounds", "bets", "sits", "void_rounds", "absorbed_rounds")
FLOAT_FIELDS = (
    "final_bankroll", "net_profit", "log_growth", "max_drawdown", "return_std", "mean_bet",
)
METRIC_KEYS = (
    "mean_terminal_bankroll", "median_terminal_bankroll", "mean_log_growth",
    "median_log_growth", "variance_log_growth", "std_log_growth", "mean_max_drawdown",
    "median_max_drawdown", "max_drawdown_q95", "participation_rate",
    "mean_wager_conditional_on_bet", "log_growth_q05", "log_growth_lower_tail_mean_05",
)


def file_hash(path):
    """Hash large archived CSVs without loading them into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values, probability):
    """Linear interpolation at (n - 1) * probability, including endpoints."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("A quantile needs data and a probability between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def percentile_interval(values):
    return [quantile(values, 0.025), quantile(values, 0.975)]


def bootstrap_indices(n, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    """One index matrix is shared by every strategy and every paired contrast."""
    if n < 2 or replicates < 2:
        raise ValueError("Bootstrap requires at least two sessions and two replications")
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(replicates)]


def paired_comparison(left, right, indices):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Paired samples must have equal lengths of at least two")
    differences = [a - b for a, b in zip(left, right)]
    estimates = [math.fsum(differences[i] for i in draw) / len(draw) for draw in indices]
    return {"estimate": statistics.mean(differences), "ci95": percentile_interval(estimates)}


def wilson_interval(count, n):
    """Two-sided 95% Wilson interval for an independent-session event rate."""
    if n < 1 or not 0 <= count <= n:
        raise ValueError("Invalid binomial observation count")
    z = statistics.NormalDist().inv_cdf(0.975)
    p = count / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def clopper_pearson_interval(count, n):
    """Exact two-sided 95% binomial interval, with analytic zero-event bounds."""
    if n < 1 or not 0 <= count <= n:
        raise ValueError("Invalid binomial observation count")
    if count == 0:
        return [0.0, 1 - 0.025 ** (1 / n)]
    if count == n:
        return [0.025 ** (1 / n), 1.0]

    def cumulative(k, probability):
        if probability == 0:
            return 1.0
        if probability == 1:
            return 0.0
        return math.fsum(
            math.comb(n, i) * probability ** i * (1 - probability) ** (n - i)
            for i in range(k + 1)
        )

    def solve(k, target):
        lower, upper = 0.0, 1.0
        for _ in range(70):
            middle = (lower + upper) / 2
            if cumulative(k, middle) > target:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / 2

    return [solve(count - 1, 0.975), solve(count, 0.025)]


def _read_csv(path, expected_fields=None):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
                or expected_fields is not None and set(reader.fieldnames) != set(expected_fields)):
            raise ValueError(f"Unexpected CSV header: {Path(path).name}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed CSV: {Path(path).name}")
    return rows


def validate_sessions(rows, manifest):
    """Reject malformed, missing or inconsistent observations; do not repair them."""
    if (manifest.get("status") != "complete" or manifest.get("mode") != "final"
            or manifest.get("strategies") != list(STRATEGIES)
            or manifest.get("rounds_per_session") != ROUNDS):
        raise ValueError("Final manifest does not describe the frozen seven-strategy study")
    seeds = manifest.get("seeds", [])
    if (len(seeds) != 300 or len(set(seeds)) != 300
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Final manifest must contain 300 distinct valid seeds")
    groups = {strategy: {} for strategy in STRATEGIES}
    for raw in rows:
        if set(raw) != set(SESSION_FIELDS):
            raise ValueError("Unexpected session columns")
        try:
            row = {**raw, **{key: int(raw[key]) for key in INTEGER_FIELDS},
                   **{key: float(raw[key]) for key in FLOAT_FIELDS}}
            finite = all(math.isfinite(row[key]) for key in FLOAT_FIELDS)
            wealth_consistent = (Decimal(str(raw["final_bankroll"]))
                                 - Decimal(str(raw["net_profit"])) == INITIAL_BANKROLL)
        except (TypeError, ValueError, InvalidOperation) as error:
            raise ValueError("Invalid numeric session observation") from error
        name, seed = row["strategy"], row["seed"]
        if name not in groups or seed not in seeds:
            raise ValueError("Unexpected session strategy or seed")
        if seed in groups[name]:
            raise ValueError("Duplicate strategy/seed session")
        if not finite:
            raise ValueError("Non-finite session observation")
        if (row["rounds"] != ROUNDS or row["bets"] + row["sits"] != ROUNDS
                or min(row[key] for key in ("bets", "sits", "void_rounds", "absorbed_rounds")) < 0
                or row["void_rounds"] + row["absorbed_rounds"] > ROUNDS):
            raise ValueError("Session opportunity or participation accounting is inconsistent")
        if (row["final_bankroll"] <= 0 or not wealth_consistent
                or not math.isclose(row["log_growth"], math.log(row["final_bankroll"] / INITIAL_BANKROLL),
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Session wealth, net profit and log growth do not reconcile")
        if (not 0 <= row["max_drawdown"] < 1 or row["return_std"] < 0
                or not 0 <= row["mean_bet"] <= 8
                or (row["bets"] == 0) != (row["mean_bet"] == 0)
                or row["bets"] > 0 and row["mean_bet"] < 1):
            raise ValueError("Invalid session risk or wager measurement")
        if row["max_drawdown"] + 1e-12 < max(0, 1 - row["final_bankroll"] / INITIAL_BANKROLL):
            raise ValueError("Session drawdown is smaller than the terminal loss from initial wealth")
        if (row["terminal_reason"] not in ("horizon", "no_positive_wager")
                or row["terminal_reason"] == "horizon" and row["absorbed_rounds"] != 0
                or row["terminal_reason"] == "no_positive_wager" and row["final_bankroll"] >= 50):
            raise ValueError("Invalid session terminal reason")
        if row["bets"] == 0 and (row["final_bankroll"] != INITIAL_BANKROLL
                or any(row[key] != 0 for key in ("net_profit", "log_growth", "max_drawdown", "return_std"))):
            raise ValueError("A zero-bet session must have unchanged wealth and zero returns and drawdown")
        if name == "never_bet" and (row["final_bankroll"] != INITIAL_BANKROLL
                or any(row[key] != 0 for key in ("net_profit", "log_growth", "max_drawdown",
                                                "return_std", "bets", "mean_bet"))):
            raise ValueError("The no-bet baseline is inconsistent")
        groups[name][seed] = row
    if (any(set(group) != set(seeds) for group in groups.values())
            or len(rows) != manifest.get("session_count") or len(rows) != 2100
            or sum(row["rounds"] for group in groups.values() for row in group.values())
            != manifest.get("opportunity_count") or manifest.get("opportunity_count") != 1050000):
        raise ValueError("Sessions are missing, unpaired or inconsistent with manifest counts")
    return {name: [group[seed] for seed in sorted(seeds)] for name, group in groups.items()}


def validate_summary(rows, groups):
    expected_columns = {
        "strategy", "sessions", "mean_log_growth", "mean_final_bankroll", "mean_net_profit",
        "mean_max_drawdown", "betting_rate", "functional_ruin_sessions",
    }
    if any(set(row) != expected_columns for row in rows):
        raise ValueError("Unexpected policy summary columns")
    if len(rows) != len(STRATEGIES) or {row["strategy"] for row in rows} != set(STRATEGIES):
        raise ValueError("Summary strategies are missing or duplicated")
    for row in rows:
        group = groups[row["strategy"]]
        expected = {
            "sessions": len(group),
            "mean_log_growth": statistics.mean(x["log_growth"] for x in group),
            "mean_final_bankroll": statistics.mean(x["final_bankroll"] for x in group),
            "mean_net_profit": statistics.mean(x["net_profit"] for x in group),
            "mean_max_drawdown": statistics.mean(x["max_drawdown"] for x in group),
            "betting_rate": sum(x["bets"] for x in group) / sum(x["rounds"] for x in group),
            "functional_ruin_sessions": sum(x["terminal_reason"] == "no_positive_wager" for x in group),
        }
        try:
            valid = all(math.isfinite(float(row[key])) and math.isclose(
                float(row[key]), value, rel_tol=1e-12, abs_tol=1e-12)
                for key, value in expected.items())
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid numeric summary observation") from error
        if not valid:
            raise ValueError(f"Session evidence does not reconcile with summary: {row['strategy']}")


def _metric_estimates(rows):
    logs = sorted(row["log_growth"] for row in rows)
    wealth = sorted(row["final_bankroll"] for row in rows)
    drawdowns = sorted(row["max_drawdown"] for row in rows)
    bets = sum(row["bets"] for row in rows)
    tail_count = max(1, math.ceil(0.05 * len(rows)))
    variance = statistics.variance(logs)
    return {
        "mean_terminal_bankroll": statistics.mean(wealth),
        "median_terminal_bankroll": statistics.median(wealth),
        "mean_log_growth": statistics.mean(logs),
        "median_log_growth": statistics.median(logs),
        "variance_log_growth": variance,
        "std_log_growth": math.sqrt(variance),
        "mean_max_drawdown": statistics.mean(drawdowns),
        "median_max_drawdown": statistics.median(drawdowns),
        "max_drawdown_q95": quantile(drawdowns, 0.95),
        "participation_rate": bets / sum(row["rounds"] for row in rows),
        "mean_wager_conditional_on_bet": (
            math.fsum(row["mean_bet"] * row["bets"] for row in rows) / bets if bets else 0.0),
        "log_growth_q05": quantile(logs, 0.05),
        "log_growth_lower_tail_mean_05": statistics.mean(logs[:tail_count]),
    }


def strategy_statistics(rows, indices):
    estimates = _metric_estimates(rows)
    draws = {key: [] for key in METRIC_KEYS}
    for indices_row in indices:
        replicate = _metric_estimates([rows[index] for index in indices_row])
        for key, value in replicate.items():
            draws[key].append(value)
    result = {key: {"estimate": value, "ci95": percentile_interval(draws[key])}
              for key, value in estimates.items()}
    n = len(rows)
    result.update({
        "sessions": n, "opportunities": sum(row["rounds"] for row in rows),
        "bets": sum(row["bets"] for row in rows), "sits": sum(row["sits"] for row in rows),
    })
    if result["bets"] == 0:
        result["mean_wager_conditional_on_bet"] = {"estimate": None, "ci95": None}
    for key, count, interval, method in (
        ("functional_ruin", sum(row["terminal_reason"] == "no_positive_wager" for row in rows),
         clopper_pearson_interval, "two-sided 95% exact Clopper-Pearson"),
        ("positive_terminal_return", sum(row["net_profit"] > 0 for row in rows),
         wilson_interval, "two-sided 95% Wilson"),
    ):
        result[key] = {"count": count, "estimate": count / n, "ci95": interval(count, n),
                       "interval_method": method}
    return result


def analyse(root=Path(".")):
    """Return deterministic derived results after checking frozen input evidence."""
    from research.verification import verify_publication

    root = Path(root)
    verify_publication(root=root)
    result_dir = root / "results"
    manifest = json.loads((result_dir / "experiment_metadata.json").read_text())
    plan = json.loads((root / "research/study_plan_dqn_extended.json").read_text())
    if set(manifest["seeds"]) != set(plan["seeds"]["evaluation"]):
        raise ValueError("Final seed identities differ from the frozen study plan")
    for name in ("sessions.csv", "summary.csv"):
        if file_hash(result_dir / name) != manifest["result_sha256"][name]:
            raise ValueError(f"Historical result checksum differs: {name}")
    groups = validate_sessions(_read_csv(result_dir / "sessions.csv", SESSION_FIELDS), manifest)
    validate_summary(_read_csv(result_dir / "summary.csv"), groups)
    indices = bootstrap_indices(len(manifest["seeds"]))
    strategies = {name: strategy_statistics(groups[name], indices) for name in STRATEGIES}
    paired = []
    primary = None
    for left, right in itertools.combinations(STRATEGIES, 2):
        is_primary = {left, right} == {"dqn", "model_based_log"}
        if is_primary:
            left, right = "dqn", "model_based_log"
        contrast = paired_comparison([row["log_growth"] for row in groups[left]],
                                     [row["log_growth"] for row in groups[right]], indices)
        contrast.update({"strategy": left, "baseline": right,
                         "role": "confirmatory_primary" if is_primary else "exploratory"})
        paired.append(contrast)
        if is_primary:
            primary = dict(contrast)
    lower, upper = primary["ci95"]
    primary.update({
        "zero_in_interval": lower <= 0 <= upper,
        "economic_geometric_ratio_difference": {
            "estimate": math.expm1(primary["estimate"]),
            "ci95": [math.expm1(lower), math.expm1(upper)],
        },
        "practical_effect_threshold": None,
        "interpretation": (
            "Selected DQN has lower mean session log growth than model-based expected-log betting"
            if upper < 0 else
            "Selected DQN has higher mean session log growth than model-based expected-log betting"
            if lower > 0 else
            "Neither superiority nor inferiority is established; this is not evidence of equivalence"
        ),
    })
    input_paths = (
        "results/experiment_metadata.json", "results/sessions.csv", "results/summary.csv",
        "research/evidence/asset_mapping.json", "research/study_plan.json",
        "research/study_plan_dqn_extended.json", "models/selection.json",
        "models/calibration_model.json", "models/dqn_primary/policy.pt",
        "models/dqn_primary/settings.json", "models/dqn_exploratory/policy.pt",
        "models/dqn_exploratory/settings.json", "research/evidence/final_evaluation_source.py.txt",
    )
    return {
        "schema_version": "blackjack_final_analysis_v1",
        "settings": {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_rng": "Python random.Random; randrange(n)",
            "bootstrap_unit": "whole independent session, common sorted seed indices across strategies",
            "confidence_level": 0.95, "bootstrap_interval": "percentile",
            "quantile_interpolation": "linear at (n-1)*p", "tail_probability": 0.05,
            "tail_mean_definition": "mean of the lowest ceil(0.05*n) session log-growth values",
            "seed_status": "Analysis-only choice: calibration bootstrap seed 25001 plus one; not prespecified",
            "method_status": "2000 paired whole-session replicates were prespecified; final RNG seed and percentile implementation were analysis choices",
            "practical_effect_threshold": None,
            "interval_scope": "pointwise; no multiple-comparison adjustment; secondary comparisons exploratory",
            "conditional_wager_note": "Bet-count-weighted mean across sessions; null when no bets occur because the conditional mean is undefined",
            "risk_note": "Drawdown is measured from settled bankroll; zero observed ruin does not establish zero true risk",
        },
        "provenance": {
            "input_sha256": {path: file_hash(root / path) for path in input_paths},
            "analysis_source_sha256": file_hash(Path(__file__)),
            "method_note": "Derived from unchanged saved final evidence; no simulations, training, model selection or final-seed reuse",
        },
        "study": {
            "strategies": list(STRATEGIES), "confirmatory_strategies": [
                name for name in STRATEGIES if name != "dqn_runner_up"],
            "exploratory_strategy": "dqn_runner_up", "session_count": len(manifest["seeds"]) * 7,
            "sessions_per_strategy": len(manifest["seeds"]), "opportunities_per_session": ROUNDS,
            "opportunity_count": manifest["opportunity_count"], "initial_bankroll": INITIAL_BANKROLL,
            "primary_dqn": "extended_3", "exploratory_dqn": "original_1",
            "pairing_note": "Common seeds match sessions, not identical realised hands when participation differs",
        },
        "validation": {
            "publication_hashes": "passed", "complete_session_pairing": "passed",
            "session_accounting": "passed", "summary_reconciliation": "passed",
            "no_bet_baseline": "passed",
            "raw_rounds": "Original raw records and their audit are preserved in the external research archive; not required to reproduce session analysis",
        },
        "strategies": strategies, "primary_comparison": primary, "paired_comparisons": paired,
    }


# American spelling is also convenient for external notebooks.
analyze = analyse


def validate_rounds(path, sessions, manifest):
    """Stream original raw evidence, checking every round and saved session metric."""
    raw_hash = file_hash(path)
    if raw_hash != manifest["result_sha256"]["rounds.csv"]:
        raise ValueError("Archived round checksum differs from the original final manifest")
    groups = validate_sessions(sessions, manifest)
    expected = {(row["strategy"], row["seed"]): row for group in groups.values() for row in group}
    current = None
    seen = set()
    statuses_by_strategy = {name: Counter() for name in STRATEGIES}
    actions_by_strategy = {name: Counter() for name in STRATEGIES}
    logs, ratios, stakes = [], [], []
    last = peak = Decimal(INITIAL_BANKROLL)
    profit = drawdown = Decimal(0)
    statuses = Counter()
    total = 0

    def finish():
        if current is None:
            return
        row = expected[current]
        comparisons = (
            (math.fsum(logs), row["log_growth"]), (float(drawdown), row["max_drawdown"]),
            (statistics.stdev(ratios), row["return_std"]),
            (statistics.mean(stakes) if stakes else 0.0, row["mean_bet"]),
        )
        if (len(logs) != ROUNDS or last != Decimal(str(row["final_bankroll"]))
                or profit != Decimal(str(row["net_profit"])) or len(stakes) != row["bets"]
                or statuses["void"] != row["void_rounds"]
                or statuses["absorbed"] != row["absorbed_rounds"]
                or any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in comparisons)):
            raise ValueError(f"Raw rounds do not reconcile to the saved session: {current}")

    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "strategy", "seed", "opportunity", "round_status", "true_count", "action_index",
            "wager", "bankroll_before", "bankroll_after", "net_profit", "round_return", "log_reward",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != required or len(reader.fieldnames) != len(required):
            raise ValueError("Unexpected raw-round columns")
        for raw in reader:
            try:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError("Malformed raw-round CSV row")
                key = raw["strategy"], int(raw["seed"])
                if key not in expected:
                    raise ValueError("Unexpected strategy/seed in raw rounds")
                if current != key:
                    finish()
                    if key in seen:
                        raise ValueError("Repeated or noncontiguous raw session")
                    seen.add(key)
                    current = key
                    logs, ratios, stakes = [], [], []
                    last = peak = Decimal(INITIAL_BANKROLL)
                    profit = drawdown = Decimal(0)
                    statuses = Counter()
                before, after = Decimal(raw["bankroll_before"]), Decimal(raw["bankroll_after"])
                gain = Decimal(raw["net_profit"])
                wager = Decimal(raw["wager"]) if raw["wager"] else Decimal(0)
                action, number = int(raw["action_index"]), int(raw["opportunity"])
                reward = float(raw["log_reward"])
                if (not all(value.is_finite() for value in (before, after, gain, wager))
                        or not math.isfinite(reward)
                        or number != len(logs) + 1 or number > ROUNDS or not 0 <= action <= 5
                        or before <= 0 or after <= 0 or before != last or after - before != gain
                        or wager != (0, 1, 2, 4, 6, 8)[action]):
                    raise ValueError("Invalid round sequence, action or bankroll accounting")
                status = raw["round_status"]
                if status not in ("complete", "void", "absorbed"):
                    raise ValueError("Unexpected round status")
                if (statuses["absorbed"] and status != "absorbed"
                        or status == "absorbed" and (wager or raw["true_count"])
                        or status != "absorbed" and not raw["true_count"]):
                    raise ValueError("Invalid absorbed sequence or missing public pre-bet count")
                if (not wager or status != "complete") and gain != 0:
                    raise ValueError("No-bet, void or absorbed round changed bankroll")
                if wager:
                    if wager > Decimal("0.02") * before or 8 * wager >= before:
                        raise ValueError("Round wager violates common exposure constraints")
                    stakes.append(float(wager))
                expected_return = float(gain / wager) if wager and status == "complete" else None
                if ((expected_return is None) != (raw["round_return"] == "")
                        or expected_return is not None and not math.isclose(
                            float(raw["round_return"]), expected_return, rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError("Invalid recorded round return")
                if not math.isclose(reward, math.log1p(float(gain / before)), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Invalid recorded log reward")
                if raw["true_count"] and not math.isfinite(float(raw["true_count"])):
                    raise ValueError("Non-finite public count")
                peak = max(peak, after)
                drawdown = max(drawdown, 1 - after / peak)
                profit += gain
                last = after
                logs.append(reward)
                ratios.append(float(gain / before))
                statuses[status] += 1
                statuses_by_strategy[key[0]][status] += 1
                actions_by_strategy[key[0]][str(action)] += 1
                total += 1
            except (KeyError, TypeError, InvalidOperation, ArithmeticError, ValueError) as error:
                raise ValueError(f"Invalid raw evidence at data row {total + 1}: {error}") from error
    finish()
    if seen != set(expected) or total != manifest["opportunity_count"]:
        raise ValueError("Raw opportunities or session identities are missing")
    return {
        "schema_version": "blackjack_raw_round_validation_v1", "status": "passed",
        "original_relative_path": "outputs/final_evaluation_7/rounds.csv",
        "input_sha256": {"rounds.csv": raw_hash,
                         "sessions.csv": manifest["result_sha256"]["sessions.csv"]},
        "session_count": len(seen), "opportunity_count": total,
        "checks": ["manifest checksum", "unique complete session opportunities", "bankroll continuity",
                   "profit and reward accounting", "initial wager grid and risk constraints",
                   "session log growth and terminal wealth", "settled-bankroll maximum drawdown",
                   "return standard deviation", "participation and conditional wager size"],
        "statuses_by_strategy": {name: dict(counts) for name, counts in statuses_by_strategy.items()},
        "actions_by_strategy": {name: dict(counts) for name, counts in actions_by_strategy.items()},
        "scope_note": "An audit of the historical raw records, not a new simulation; session analysis reproduces without this archived CSV",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rounds", type=Path, help="Optional original archived rounds.csv for a separate full audit")
    args = parser.parse_args()
    result = analyse(args.root)
    if args.rounds is not None:
        manifest = json.loads((args.root / "results/experiment_metadata.json").read_text())
        raw_report = validate_rounds(args.rounds,
                                    _read_csv(args.root / "results/sessions.csv", SESSION_FIELDS), manifest)
        print(json.dumps(raw_report, indent=2, sort_keys=True, allow_nan=False))
    primary = result["primary_comparison"]
    print(f"Validated {result['study']['session_count']} saved sessions; "
          f"primary mean {primary['estimate']:+.12f}, 95% CI {primary['ci95']}")


if __name__ == "__main__":
    main()
