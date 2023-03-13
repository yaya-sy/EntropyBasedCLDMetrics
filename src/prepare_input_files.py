"""Module for preparing input files for the model."""
from typing import Union, List
from math import log
from pathlib import Path
from argparse import ArgumentParser
from itertools import tee
import re
import logging
import phonemizer
import kenlm
import h5py
import soundfile as sf
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def wavform(audio_path: Path):
    """Reads audio as waveform."""
    return sf.read(audio_path)

def h5_dataset(utterances_paths: Path, audio_folder: Path, output_folder: Path):
    """Creates h5py data for the dataloader."""
    output_folder.mkdir(exist_ok=True, parents=True)
    f = h5py.File(output_folder / f"{utterances_paths.stem}.hdf5", 'w')
    already_opened = dict()
    sorted_utterances = []
    LOGGER.info("Creating h5 dataset...")
    with open(utterances_paths, "r") as audio_paths:
        copied_audio_paths, audio_paths = tee(audio_paths, 2)
        total = sum(1 for _ in copied_audio_paths)
        for line in tqdm(audio_paths, total=total):
            informations = line.strip().split("\t")
            utterance_id, audio_path, *timemarks = informations
            audio_path = audio_folder / audio_path
            if audio_path in already_opened:
                audio, sr = already_opened[audio_path]
            else:
                already_opened.clear()
                audio, sr = wavform(audio_path)
                already_opened[audio_path] = (audio, sr)
            if timemarks:
                raw_onset, raw_offset = timemarks
                onset = int(raw_onset) / 1000
                offset = int(raw_offset) / 1000
                onset = int(onset * sr)
                offset = int(offset * sr)
                utterance = audio[onset:offset]
            else:
                utterance = audio
            sorted_utterances.append((audio.shape[0], utterance_id))
            f.create_dataset(utterance_id, data=utterance)
    sorted_utterances = sorted(sorted_utterances)
    _, ids = zip(*sorted_utterances)
    with open(output_folder / f"{utterances_paths.stem}.sorted", 'w') as sorted_paths_file:
        for id in ids:
            sorted_paths_file.write(f"{id}\n")

def compute_entropy(model: kenlm.Model, utterance: Union[str, List[str]]) -> float:
    """Computes the entropy of a given utterance from a given model."""
    ppl = model.perplexity(utterance)
    return log(ppl)

def entropies_file(utterances_segments: Path,
                   ngram_model_path: str,
                   output_folder: Path) -> None:
    """Creates targets (entropies)."""
    model = kenlm.Model(ngram_model_path)
    separator = phonemizer.separator.Separator(phone=' ', word='  ')
    LOGGER.info("Creating targets file (entropies)...")
    with open(utterances_segments, "r") as text_file:
        ids, utterances = zip(*[line.strip().split("\t") for line in text_file])
        phonemized = phonemizer.phonemize(text=utterances,
                                          strip=True,
                                          preserve_empty_lines=True,
                                          language_switch="remove-flags",
                                          separator=separator)
    tokenized = [re.sub(" +", " ", utterance) for utterance in phonemized]
    entropies = [compute_entropy(model, utterance) for utterance in tokenized]
    with open(output_folder / f"{utterances_segments.stem}.entropies", "w") as entropies_file:
        for id, entropy in zip(ids, entropies):
            entropies_file.write(f"{id}\t{entropy}\n")

def main():
    parser = ArgumentParser()
    parser.add_argument("-u", "--utterances_paths",
                        help="File mapping each utterance id to its audio path and frames.",
                        required=True)
    parser.add_argument("-s", "--utterances_segments",
                        help="File mapping each utterance id to its orthogaphic transcription.",
                        required=True)
    parser.add_argument("-a", "--audio_folder",
                        help="Folder where raw recordings are stored.",
                        required=True)
    parser.add_argument("-o", "--output_folder",
                        help="Where the output files will be saved.",
                        required=True)
    parser.add_argument("-m", "--ngram_model",
                        help="The path to the ngram model.",
                        required=True)

    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    utterances_paths = Path(args.utterances_paths)
    utterances_paths.rename(output_folder / f"{utterances_paths.stem}.paths")
    h5_dataset(output_folder / f"{utterances_paths.stem}.paths", Path(args.audio_folder), output_folder)
    entropies_file(Path(args.utterances_segments), args.ngram_model, output_folder)

if __name__ == "__main__":
    main()