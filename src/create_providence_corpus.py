from pathlib import Path
from argparse import ArgumentParser
import string
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from tqdm import tqdm
import pylangacq

PUNCTS = "".join(set(string.punctuation) - {"'"})

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

        orthographic_folder = output_folder / "orthographic" / child_name / raw_age
        orthographic_folder.mkdir(exist_ok=True, parents=True)
        with open(orthographic_folder / f"{speaker}.orthographic", "w") as orthographic_file:
            orthographic_file.write("\n".join(data["transcription"]))

        cleaned_folder = output_folder / "cleaned" / child_name / raw_age
        cleaned_folder.mkdir(exist_ok=True, parents=True)
        cleaned = [utterance.translate(str.maketrans('', '', PUNCTS)) for utterance in data["clean_transcription"]]
        with open(cleaned_folder / f"{speaker}.cleaned", "w") as orthographic_file:
            orthographic_file.write("\n".join(cleaned))

        timemarks_folder = output_folder / "timemarks" / child_name / raw_age
        timemarks_folder.mkdir(exist_ok=True, parents=True)
        timemarks = zip(data["segment_onset"], data["segment_offset"])
        timemarks = ["\t".join((str(timemark[0]), str(timemark[1]))) for timemark in timemarks]
        with open(timemarks_folder / f"{speaker}.timemarks", "w") as orthographic_file:
            orthographic_file.write("\n".join(timemarks))

        with open(orthographic_folder / f"months.txt", "w") as months_orthographic:
            months_orthographic.write(months) 
        with open(cleaned_folder / f"months.txt", "w") as months_cleaned:
            months_cleaned.write(months) 
        with open(timemarks_folder / f"months.txt", "w") as months_timemarks:
            months_timemarks.write(months) 

        with open(orthographic_folder / f"filename.txt", "w") as filenamne_orthographic:
            filenamne_orthographic.write(filename) 
        with open(cleaned_folder / f"filename.txt", "w") as filename_cleaned:
            filename_cleaned.write(filename) 
        with open(timemarks_folder / f"filename.txt", "w") as filename_timemarks:
            filename_timemarks.write(filename) 

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_csv", help="CSV file containing all the files.")
    parser.add_argument("-c", "--cha_folder", help="Folder containing the cha files.")
    parser.add_argument("-o", "--output_folder", help="CSV file containing all the files.")

    args = parser.parse_args()
    dataframe = pd.read_csv(args.input_csv, low_memory=False)
    dataframe = dataframe.groupby(["raw_filename", "speaker_role"])
    create_folders(dataframe, Path(args.output_folder), Path(args.cha_folder) / "annotations" / "cha" / "raw")

if __name__ == "__main__":
    main()