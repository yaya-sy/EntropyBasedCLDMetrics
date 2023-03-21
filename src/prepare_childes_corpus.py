"""Module for preparing the childes corpora for model training/testing."""
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

def prepare(childes_folder: Path, output_folder: Path):
    """
    Prepares the input files for a given childes corpus.
    It will create the utterances file, their audio paths\
    and the informations (i.e: for each utterance, who speaks\
    and what age.)
    """
    output_filename = childes_folder.stem
    children = list(childes_folder.glob("cleaned/*/"))
    utterances_paths = [] # will contain all the audio path, and the onsets/offsets corresponding to each utterance.
    utterances_segments = [] # will map each audio utterance to its orthographic version.
    utterances_informations = [] # will contain the family, speaker, age, for each produced utterance.

    for child in tqdm(children):
        child_name = child.stem
        ages = list(child.glob("*/"))
        for age in tqdm(ages):
            with open(age / "months.txt") as months_file:
                months = next(months_file)
            with open(age / "filename.txt") as filename_file:
                filename = next(filename_file)
            age_utterances = []
            for speaker_utterances in age.glob("*.cleaned"):
                with open(speaker_utterances, "r") as utterances:
                    target_speaker = speaker_utterances.stem
                    with open(childes_folder / "timemarks" / child_name / age.stem / f"{target_speaker}.timemarks") as timemarks:
                        utterances, timemarks = utterances.readlines(), timemarks.readlines()
                        assert len(utterances) == len(timemarks), "Mismatch between timemarks and utterances"
                        for utterance, timemark in zip(utterances, timemarks):
                            age_utterances.append((target_speaker, utterance, timemark))
            for idx, (target_speaker, utterance, timemark) in enumerate(age_utterances):
                utterance_id = f"{filename}_{target_speaker}_{idx:05d}"
                onset, offset = timemark.strip().split("\t")
                orthographic = utterance.strip()
                # Important logic here:
                # a) For training (thomas), we necessary need transcription (entropy) and speech pairs.
                # So we have to keep only utterances for which we have transcription AND speech.
                # b) For testing (providence), we may have speech without transcription (incomprehensible speech).
                # In this case, we will keep the speech for predicting the entropy even if this speech is not transcribed.
                if not orthographic and childes_folder.stem == "Thomas":
                    continue
                path = f"{child_name}/{filename}" if len(children) > 1 else filename
                utterances_paths.append(f"{utterance_id}\t{path}.wav\t{onset}\t{offset}")
                utterances_segments.append(f"{utterance_id}\t{orthographic}")
                utterances_informations.append(f"{utterance_id}\t{child_name}\t{target_speaker}\t{months}")
    with open(output_folder / f"{output_filename}.paths", "w") as paths_file:
        paths_file.write("\n".join(utterances_paths))
    with open(output_folder / f"{output_filename}.segments", "w") as segments_file:
        segments_file.write("\n".join(utterances_segments))
    with open(output_folder / f"{output_filename}.infos", "w") as infos_file:
        infos_file.write("\n".join(utterances_informations))

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_folder",
                        help="The folder containing the providence data.",
                        type=str)
    
    args = parser.parse_args()
    output_folder = Path(args.input_folder) / "model_inputs"
    output_folder.mkdir(exist_ok=True, parents=True)
    prepare(Path(args.input_folder), output_folder)
if __name__ == "__main__":
    main()