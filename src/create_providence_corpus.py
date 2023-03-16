from pathlib import Path
import string
import re
from argparse import ArgumentParser
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from tqdm import tqdm
import pylangacq

PUNCTS = "".join(set(string.punctuation) - {"'"})

def write_utterances(utterances: str, output_file: Path) -> None:
    """Writes utterances in folder with one utterance by file."""
    with open(output_file, "w") as utterance_file:
        for utterance in utterances:
            if output_file.suffix == ".cleaned":
                utterance = utterance.translate(str.maketrans('', '', PUNCTS))
                utterance = re.sub(' +', ' ', utterance)
            utterance_file.write(f"{utterance}\n")

def create_folders(groups: DataFrameGroupBy, output_folder: Path, cha_folder: Path) -> None:
    allowed_speakers = {"Mother", "Target_Child"}
    for group, data in tqdm(groups):
        cha_file, speaker = group
        cha_file = cha_folder / cha_file
        cha = pylangacq.read_chat(str(cha_file))
        months = cha.ages(months=True)[0]
        if speaker not in allowed_speakers:
            continue
        months = str(months)
        child_name = cha_file.parent.stem
        filename = cha_file.stem
        raw_age = filename.split("_")[-1]
        assert len(data["transcription"]) == len(data["clean_transcription"]), "Some mismatches between non-cleaned and cleaned."

        output_folder_orthographic = output_folder / "orthographic" / child_name / raw_age
        output_folder_cleaned = output_folder / "cleaned" / child_name / raw_age
        output_folder_timemarks = output_folder / "timemarks" / child_name / raw_age

        output_folder_orthographic.mkdir(exist_ok=True, parents=True)
        output_folder_cleaned.mkdir(exist_ok=True, parents=True)
        output_folder_timemarks.mkdir(exist_ok=True, parents=True)

        write_utterances(data["transcription"], output_folder_orthographic / f"{speaker}.orthographic")
        write_utterances(data["clean_transcription"], output_folder_cleaned / f"{speaker}.cleaned")
        timemarks = zip(data["segment_onset"], data["segment_offset"])
        timemarks = ["\t".join((str(timemark[0]), str(timemark[1]))) for timemark in timemarks]
        write_utterances(timemarks, output_folder_timemarks / f"{speaker}.timemarks")

        with open(output_folder_orthographic / f"months.txt", "w") as months_orthographic:
            months_orthographic.write(months) 
        with open(output_folder_cleaned / f"months.txt", "w") as months_cleaned:
            months_cleaned.write(months) 
        with open(output_folder_timemarks / f"months.txt", "w") as months_timemarks:
            months_timemarks.write(months) 

        with open(output_folder_orthographic / f"filename.txt", "w") as filenamne_orthographic:
            filenamne_orthographic.write(filename) 
        with open(output_folder_cleaned / f"filename.txt", "w") as filename_cleaned:
            filename_cleaned.write(filename) 
        with open(output_folder_timemarks / f"filename.txt", "w") as filename_timemarks:
            filename_timemarks.write(filename) 

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_csv", help="CSV file containing all the files.")
    parser.add_argument("-c", "--childes_corpus", help="Folder containing the childes corpus.")
    parser.add_argument("-o", "--output_folder", help="CSV file containing all the files.")

    args = parser.parse_args()
    dataframe = pd.read_csv(args.input_csv, low_memory=False)
    dataframe = dataframe.groupby(["raw_filename", "speaker_role"])
    create_folders(dataframe,
                   Path(args.output_folder),
                   Path(args.childes_corpus) / "annotations" / "cha" / "raw")
    # /scratch2/whavard/DATA/LSFER/providence/segments_clean_to_synthetise.csv

if __name__ == "__main__":
    main()