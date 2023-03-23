"""Module for creating hierarchical data organization of the thomas corpus."""
from utterances_cleaner_thomas import UtterancesCleaner
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import pandas as pd
import pylangacq
from tqdm import tqdm

CLEANER = UtterancesCleaner("extra/markers.json")

def get_data(cha_folder: Path) -> dict:
    """Retrieves all the relevant data from the csv files."""
    converted_folder = (cha_folder / "converted")
    csvs = list(converted_folder.glob("*.csv"))
    for csv in tqdm(csvs):
        dataframe = pd.read_csv(csv)
        needed_columns = zip(dataframe["transcription"], dataframe["segment_onset"],
                             dataframe["segment_offset"], dataframe["speaker_role"])
        ages = set(dataframe["raw_filename"])
        assert len(ages) == 1, f"Need to be one file per age. Instead, the csv {csv} has {len(ages)} ages."
        filename = list(ages)[0]
        cha_file = cha_folder / "raw" / filename
        cha = pylangacq.read_chat(str(cha_file))
        age = cha.ages(months=True)[0]
        data = {"age" : age, "raw_age": cha_file.stem, "data": defaultdict(list)}
        for utterance_raw, onset, offset, speaker_role in needed_columns:
            utterance = pylangacq.chat._clean_utterance(utterance_raw)
            cleaned = CLEANER.clean(utterance)
            data["data"][speaker_role].append((utterance_raw, cleaned, onset, offset))
        yield data

def write_utterances(utterances: str, output_file: Path) -> None:
    """Writes utterances in a given file."""
    with open(output_file, "w") as utterance_file:
        for utterance in utterances:
            utterance_file.write(f"{utterance}\n")

def make_folder(cha_folder: Path, output_folder: Path):
    """
    Creates the folders for the different utterances types:\
    orthographic, cleaned, timemarks.
    """
    child_name = "Thomas"
    allowed_speakers = {"Mother", "Target_Child"}
    for data in get_data(cha_folder):
        age_orthographic_folder = output_folder / "orthographic" / child_name / data["raw_age"]
        age_orthographic_folder.mkdir(exist_ok=True, parents=True)
        age_cleaned_folder = output_folder / "cleaned" / child_name / data["raw_age"]
        age_cleaned_folder.mkdir(exist_ok=True, parents=True)
        age_timemarks_folder = output_folder / "timemarks" / child_name / data["raw_age"]
        age_timemarks_folder.mkdir(exist_ok=True, parents=True)

        for speaker_role in data["data"]:
            speaker_data = list(zip(*data["data"][speaker_role]))
            utterances, cleaneds, onsets, offsets = speaker_data
            timemarks = [f"{onset}\t{offset}" for onset, offset in zip(onsets, offsets)]
            if speaker_role not in allowed_speakers:
                continue
            assert len(utterances) == len(cleaneds) == len(timemarks), "Mismatch in the data"

            utterance_orthographic_output = age_orthographic_folder / f"{speaker_role}.orthographic"
            utterance_cleaned_output = age_cleaned_folder / f"{speaker_role}.cleaned"
            utterance_timemarks_output = age_timemarks_folder / f"{speaker_role}.timemarks"
        
            write_utterances(utterances, utterance_orthographic_output)
            write_utterances(cleaneds, utterance_cleaned_output)
            write_utterances(timemarks, utterance_timemarks_output)
        
        with open(age_orthographic_folder / "months.txt", "w") as months_file:
            months_file.write(str(data["age"]))
        with open(age_orthographic_folder / "filename.txt", "w") as filename_file:
            filename_file.write(str(data["raw_age"]))

        with open(age_cleaned_folder / "months.txt", "w") as months_file:
            months_file.write(str(data["age"]))
        with open(age_cleaned_folder / "filename.txt", "w") as filename_file:
            filename_file.write(str(data["raw_age"]))

        with open(age_timemarks_folder / "months.txt", "w") as months_file:
            months_file.write(str(data["age"]))
        with open(age_timemarks_folder / "filename.txt", "w") as filename_file:
            filename_file.write(str(data["raw_age"]))

def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--childes_corpus",
                        help="Folder containing childes corpus.",
                        required=True)
    parser.add_argument("-o", "--output_folder",
                        help="Where the folder will be stored",
                        required=True)

    args = parser.parse_args()
    make_folder(Path(args.childes_corpus) / "annotations/cha", Path(args.output_folder))

if __name__ == "__main__":
    main()