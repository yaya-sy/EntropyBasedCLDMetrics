"""Module for preapring the csv file for statistcal analysis and visualization."""
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
    parser.add_argument("-t", "--informations_file",
                        help="The text file containing the informations about\
                            who produced the utterance at what age, etc.")
    parser.add_argument("-o", "--output_folder",
                        help="The folder to where store the output csv.")
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    results = get_df(args.input_csv, args.informations_file)
    results.to_csv(output_folder / f"{args.input_csv}_analysis.csv")

if __name__ == "__main__":
    main()
