"""Module for preparing the librispeech data\
    in order to train the ngram language model."""
from typing import Union
from pathlib import Path
import re
from argparse import ArgumentParser
import logging
from tqdm import tqdm
import phonemizer

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def get_utterances(input_folder: Path, output_folder: Path) -> None:
    """Creating the file containing librispeech utterances."""
    LOGGER.info("Getting utterances...")
    transcriptions = list(input_folder.rglob("*.trans.txt"))
    with open(output_folder / "librispeech.orthographic", "w") as output_file:
        for transcription in tqdm(transcriptions):
            with open(transcription, "r") as transcription_file:
                for line in transcription_file:
                    utterance_id = re.findall(r"\d+\-\d+\-\d+", line)[0]
                    line = re.sub(f"{utterance_id} ", "", line)
                    line = line.lower().strip()
                    output_file.write(f"{line}\n")

def phonemize(inpu_file: Union[Path, str], output_folder: Path) -> None:
    """Mapping each orthographic utterance to its phonemic version."""
    separator = phonemizer.separator.Separator(phone=' ', word='  ')
    with open(inpu_file, "r") as transcription_file:
        utterances = [line.strip() for line in transcription_file]
    LOGGER.info("Phonemizing...")
    phonemized = phonemizer.phonemize(text=utterances,
                                      strip=True,
                                      preserve_empty_lines=True,
                                      language_switch="remove-flags",
                                      separator=separator)
    tokenized = [re.sub(" +", " ", utterance) for utterance in phonemized]
    with open(output_folder / "librispeech.phonemized", "w") as output_file:
        output_file.write("\n".join(tokenized))

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_folder",
                        help="Folder of the librispeech corpus.",
                        required=False)
    parser.add_argument("-o", "--output_folder",
                        help="Where the output files will be saved.",
                        required=True)

    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    if args.input_folder is not None:
        get_utterances(Path(args.input_folder), output_folder)
    phonemize("data/ngram_lm/librispeech.orthographic", output_folder)
if __name__ == "__main__":
    main()
    # phonemize(Path("/scratch1/data/raw_data/LibriSpeech/train-clean-360/"), Path("data/training"))