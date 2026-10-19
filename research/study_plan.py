
import hashlib
import json
from pathlib import Path


SEED_ROLES = ("calibration", "calibration_validation", "train_1", "train_2",
              "train_3", "selection", "evaluation")  # 7 categories of research data we need, each with own seeds


def plan_hash(plan):
    # identify the complete agreed research settings using a stable hash
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


# check the plan is valid
def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Invalid research plan")

    # check that every research stage has a seed allocation
    roles = plan.get("seeds")
    if not isinstance(roles, dict) or set(roles) != set(SEED_ROLES):
        raise ValueError("Research plan must contain every seed role")

    # track seeds we have already encountered
    seen = set()
    for role, seeds in roles.items():

        # each research stage must have a nonempty list of seeds
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Missing seeds for {role}")

        for seed in seeds:

            # seeds must be nonnegative integers
            if type(seed) is not int or seed < 0:
                raise ValueError("Seeds must be nonnegative integers")

            # stop different stages from using the same seed
            if seed in seen:
                raise ValueError("Research seed roles must not overlap")

            # remember this seed so we can detect future duplicates
            seen.add(seed)

    # get the calibration settings from the plan
    settings = plan.get("calibration")

    # make sure all required settings are present
    if not isinstance(settings, dict) or set(settings) != {
        "rounds_per_session", "bins", "min_samples", "min_sessions",
        "bootstrap_replicates", "bootstrap_seed", "max_ci_width", "min_variance"
    }:
        raise ValueError("Incomplete calibration settings")

    # check that sample sizes and bootstrap counts are valid integers
    for name in ("rounds_per_session", "min_samples", "min_sessions", "bootstrap_replicates"):
        if type(settings[name]) is not int or settings[name] < (100 if name == "bootstrap_replicates" else 1):
            raise ValueError(f"Invalid {name}")

    # check that the bootstrap random seed is valid
    if type(settings["bootstrap_seed"]) is not int or settings["bootstrap_seed"] < 0:
        raise ValueError("Invalid bootstrap seed")

    # reuse our existing functions to validate numerical settings
    from research.calibration_model import parse_bins
    from strategies.betting import decimal_value

    parse_bins(settings["bins"])

    # confidence interval width and minimum variance must be positive
    if decimal_value(settings["max_ci_width"], "max_ci_width") <= 0:
        raise ValueError("Confidence interval width must be positive")
    if decimal_value(settings["min_variance"], "min_variance") <= 0:
        raise ValueError("Minimum variance must be positive")

    # return the validated research plan
    return plan


def source_hashes():
    # record the exact simulator and calibration code used to create research data
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "blackjack").glob("*.py"))
    paths += [root / "strategies" / name for name in
              ("basic.py", "betting.py", "flat.py", "hi_lo.py", "runner.py")]
    paths += [root / "research" / name for name in
              ("study_plan.py", "calibration.py", "calibration_model.py", "validate_calibration.py")]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def load_plan(path):
    # load the recorded plan, no experiments can invent seeds automatically
    plan = json.loads(Path(path).read_text())
    return validate_plan(plan)
