from pathlib import Path
from argparse import ArgumentParser
from itertools import tee
import re
from tqdm import tqdm

def prepare(input_folder: Path, output_folder: Path):
    """Creates files for """
    utterances_paths = []
    utterances_segments = []
    transcriptions = list(input_folder.rglob("*.trans.txt"))
    for transcription in tqdm(transcriptions):
        speech_folder = transcription.parent
        with open(transcription, "r") as transcription_file:
            for line in transcription_file:
                utterance_id = re.findall(r"\d+\-\d+\-\d+", line)[0]
                line = re.sub(f"{utterance_id} ", "", line)
                line = line.lower().strip()
                utterances_paths.append(f"{utterance_id}\t{speech_folder / utterance_id}.flac")
                utterances_segments.append(f"{utterance_id}\t{line}")
    
    with open(output_folder / f"librispeech.paths", "w") as paths_file:
        paths_file.write("\n".join(utterances_paths))
    with open(output_folder / f"librispeech.segments", "w") as segments_file:
        segments_file.write("\n".join(utterances_segments))

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_folder",
                        help="Folder of the librispeech corpus.",
                        required=True)
    parser.add_argument("-o", "--output_folder",
                        help="Where the output files will be saved.",
                        required=True)

    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    
    prepare(Path(args.input_folder), output_folder)
    
if __name__ == "__main__":
    main()
# /scratch1/data/raw_data/LibriSpeech/train-clean-100/