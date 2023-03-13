
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

def prepare(childes_folder: Path, output_folder: Path, output_filename="providence"):
    """Prepares the input files for the a given childes corpus."""
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
            for idx, utterance in enumerate(age.glob("**/*.cleaned")):
                target_speaker = utterance.parent.stem
                utterance_id = f"{filename}_{target_speaker}_{idx:05d}"
                utterance_path = Path("/".join(utterance.parts[-4:]))
                utterance_stem = utterance_path.stem
                with open(childes_folder / "timemarks" / utterance_path.parent / f"{utterance_stem}.timemarks") as timemarks_file:
                    timemarks = next(timemarks_file)
                    onset, offset = timemarks.split("\t")
                with open(utterance) as utterance_file:
                    orthographic = utterance_file.readline().strip()
                if not orthographic:
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
    parser.add_argument("-o", "--output_folder",
                        help="The folder where the output files will be stored.",
                        type=str)
    
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    prepare(Path(args.input_folder), output_folder)
if __name__ == "__main__":
    main()