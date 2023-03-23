"""Module for preparing the CSVs results (expect HuBERT experiments)\
    for plotting and modelling."""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

def get_df(results_csv: str, informations_file: str):
    """Re-create DataFrame results with informations about families, ages, speakers."""
    results_df = pd.read_csv(results_csv, index_col=0)
    informations_df = []
    with open(informations_file, "r") as informations:
        for line in informations:
            utterance_id, child, speaker, age = line.split("\t")
            informations_df.append({
                "utterance_id": utterance_id,
                "family": child,
                "speaker": speaker,
                "age": float(age)
             })
    informations_df = pd.DataFrame(informations_df, index=None)
    results_df = results_df.merge(informations_df, on="utterance_id")
    results_df = results_df.groupby(["family", "speaker", "age"]).mean()
    return results_df

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_csv",
                        help="The CSV containing the predited entropies.")
    args = parser.parse_args()
    output_folder = Path("results")
    output_folder.mkdir(exist_ok=True, parents=True)
    output_filename = Path(args.input_csv).stem
    results = get_df(args.input_csv, "data/Providence/model_inputs/Providence.infos")
    results.to_csv(output_folder / f"{output_filename}_analysis.csv")

if __name__ == "__main__":
    main()
