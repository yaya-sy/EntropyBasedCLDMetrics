from .utterances_cleaner_thomas import UtterancesCleaner
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import pylangacq
from tqdm import tqdm

CLEANER = UtterancesCleaner("extra/markers.json")

def get_data(cha_folder: Path):
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
        data = {"age" : age, "raw_age": cha_file.stem, "data": []}
        for utterance, onset, offset, speaker_role in needed_columns:
            utterance = pylangacq.chat._clean_utterance(utterance)
            cleaned = CLEANER.clean(utterance)
            data["data"].append((speaker_role, utterance, cleaned, onset, offset))
        yield data

def write_utterances(utterance: str, folder: Path, suffix: str, idx: int) -> None:
    """Writes utterances in folder with one utterance by file."""
    folder.mkdir(exist_ok=True, parents=False)
    with open(folder / f"utterance_{idx:04d}.{suffix}", "w") as utterance_file:
        utterance_file.write(utterance)

def make_folder(cha_folder: Path, output_folder: Path, child_name: str):
    """
    Creates the folders for the different utterances types:\
    orthographic, cleaned, timemarks.
    Due to the many folder and file creations,\
    this way of doing takes some times but it's good\
    for the readability and the transparency of the different\
    stages.
    """
    
    allowed_speakers = {"Mother", "Target_Child"}
    for data in get_data(cha_folder):
        age_orthographic_folder = output_folder / "orthographic" / child_name / data["raw_age"]
        age_orthographic_folder.mkdir(exist_ok=True, parents=True)
        age_cleaned_folder = output_folder / "cleaned" / child_name / data["raw_age"]
        age_cleaned_folder.mkdir(exist_ok=True, parents=True)
        age_timemarks_folder = output_folder / "timemarks" / child_name / data["raw_age"]
        age_timemarks_folder.mkdir(exist_ok=True, parents=True)

        for idx, (speaker_role, utterance, cleaned, onset, offset) in enumerate(data["data"]):
            if speaker_role not in allowed_speakers:
                continue
            utterance_orthographic_output = age_orthographic_folder / speaker_role
            utterance_cleaned_output = age_cleaned_folder / speaker_role
            utterance_timemarks_output = age_timemarks_folder / speaker_role
        
            write_utterances(utterance, utterance_orthographic_output, "orthographic", idx)
            write_utterances(cleaned, utterance_cleaned_output, "cleaned", idx)
            write_utterances(f"{onset}\t{offset}", utterance_timemarks_output, "timemarks", idx)
        
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
    parser.add_argument("-c", "--cha_folder",
                        help="Folder containing the cha annotations.")
    parser.add_argument("-n", "--child_name",
                        help="The name of the child.")
    parser.add_argument("-o", "--output_folder",
                        help="Where the folder will be stored")

    args = parser.parse_args()
    make_folder(Path(args.cha_folder), Path(args.output_folder), args.child_name)

if __name__ == "__main__":
    main()