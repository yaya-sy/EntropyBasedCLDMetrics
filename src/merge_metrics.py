"""Module that merge the results CSVs with the standard child development metrics."""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import pylangacq
from tqdm import tqdm

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--csv_results",
                        help="The csv already prepared for analysis")
    parser.add_argument("-c", "--childes_providence",
                        required=True,
                        help="Path to childes providence corpus.")

    return parser.parse_args()

def get_families(metrics_csv, base_path):
    """Prepares the kideval csv by adding a family column\
       and recomputing the ages."""
    families = []
    for filename in tqdm(metrics_csv["File"]):
        if isinstance(filename, float):
            families.append(float("nan"))
            continue
        families.append(filename.split("/")[0])
        input_file = base_path / filename
        cha = pylangacq.read_chat(str(input_file))
        age = cha.ages(months=True)[0]
        if age == 0.0:
            continue
        metrics_csv.loc[(metrics_csv["File"] == filename), "age"] = age
    metrics_csv["family"] = families
    return metrics_csv

def merge(metrics, results, output_filename):
    results = results.merge(metrics, on=['family', 'age'])
    results.to_csv(f"results/{output_filename}.csv")

def main():
    args = get_args()

    metrics_csv = pd.read_csv("extra/chi.kideval.csv", sep=";")
    metrics_csv = metrics_csv.rename(columns={"Age(Month)": "age"})
    metrics_csv = get_families(metrics_csv, Path(args.childes_providence) / "annotations" / "cha" / "raw")

    csv_results = pd.read_csv(args.csv_results)
    csv_results = csv_results.loc[csv_results["speaker"] == "Target_Child"]

    merge(metrics_csv, csv_results, f"Metrics_{Path(args.csv_results).stem}")

if __name__ == "__main__":
    main()