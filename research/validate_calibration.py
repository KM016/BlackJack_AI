import argparse
import json
from decimal import Decimal
from pathlib import Path

from research.calibration_model import CalibrationModel, read_samples, summarise
from research.study_plan import load_plan, plan_hash, validate_plan

# take the results from calibration and see how well its estimates agree with a seperate set of bj simulations

# evaluate a frozen calibration model using independent validation sessions
def validate_model(model_path, validation_folder, plan, output):

    # check the research plan and load our fitted calibration model
    validate_plan(plan)
    model = CalibrationModel.load(model_path)

    # check that the model uses the correct research plan and blackjack rules
    fitted = model.get_document()
    if fitted["plan_hash"] != plan_hash(plan):
        raise ValueError("Model does not belong to this research plan")
    if fitted["population"] != "two_seats_fixed_one_unit_s17_das":
        raise ValueError("Calibration population does not match")

    # load the validation data and check it matches our calibration setup
    metadata, rows = read_samples(validation_folder, "calibration_validation", plan)
    if metadata["population"] != fitted["population"]:
        raise ValueError("Calibration and validation populations differ")
    if metadata["code_hashes"] != fitted["metadata"]["code_hashes"]:
        raise ValueError("Calibration and validation used different code")

    # make sure calibration and validation used different random seeds
    if set(metadata["seeds"]).intersection(fitted["metadata"]["seeds"]):
        raise ValueError("Calibration and validation seeds overlap")

    # calculate the mean, variance and confidence intervals from validation data
    summary = summarise(rows, metadata["seeds"], "calibration_validation", plan["calibration"])

    # compare the calibration and validation results for each true count bin
    comparisons = []
    for fitted_bin, validation_bin in zip(fitted["summary"]["bins"], summary["bins"]):

        # make sure we are comparing the same true count ranges
        if fitted_bin["label"] != validation_bin["label"]:
            raise ValueError("True-count bins differ between data sets")

        # get the calibration mean and validation confidence interval
        calibration_mean = fitted_bin["mean"]
        validation_ci = validation_bin["mean_ci"]

        # record the results and check whether the calibration mean falls inside the validation interval
        comparisons.append({
            "bin": fitted_bin["label"],
            "calibration_count": fitted_bin["count"],
            "validation_count": validation_bin["count"],
            "calibration_mean": calibration_mean,
            "validation_mean": validation_bin["mean"],
            "calibration_variance": fitted_bin["variance"],
            "validation_variance": validation_bin["variance"],
            "validation_mean_ci": validation_ci,
            "calibration_mean_inside_validation_ci": (
                Decimal(validation_ci[0]) <= Decimal(calibration_mean) <= Decimal(validation_ci[1])
            ) if calibration_mean is not None and validation_ci is not None else None,
            "calibration_supported": fitted_bin["supported"],
            "validation_supported": validation_bin["supported"]
        })

    # create a report containing our validation results and limitations
    report = {
        "version": "calibration_validation_v1", "plan_hash": plan_hash(plan),
        "calibration_seed_count": len(fitted["metadata"]["seeds"]),
        "validation_seed_count": len(metadata["seeds"]),
        "validation_complete_rounds": summary["complete_rounds"],
        "validation_void_rounds": summary["void_rounds"],
        "comparisons": comparisons,
        "limitations": [
            "Pointwise 95% confidence intervals are not simultaneous guarantees.",
            "Calibration and validation sessions are separate; rounds within a session are dependent.",
            "True-count bins omit exact shoe composition and other sources of state variation.",
            "This validation is not a profitability evaluation of a Kelly strategy."
        ]
    }

    # save the validation report as a JSON file without overwriting existing results
    with Path(output).open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    # return the report so other files can use the results
    return report

# run calibration validation from the command line
def main():

    # define the arguments needed to run validation
    parser = argparse.ArgumentParser(description="Validate blackjack calibration on independent sessions")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # load the research plan and validate our fitted model
    plan = load_plan(args.plan)
    report = validate_model(args.model, args.data, plan, args.output)
    
    # print the number of completed validation rounds
    print(f"Validated on {report['validation_complete_rounds']} completed rounds")


if __name__ == "__main__":
    main()
