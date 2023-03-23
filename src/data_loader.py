"""This module implements a dataloader for the model."""
from typing import Union, Iterator, Tuple, List, Optional
from pathlib import Path
from itertools import islice, chain
import logging
import random
from random import shuffle
import torch
from transformers import AutoProcessor
import numpy as np
import h5py
from tqdm import tqdm

DataPath = Union[str, Path]
Array = Union[np.ndarray, torch.Tensor]
UtteranceId = str
Utterance = Array
Target = float
DataItem = Tuple[UtteranceId, Utterance, Target]

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(1797)
random.seed(1797)

class DataLoader:
    """
    A dataloder for the entropies predictor model.

    Parameters
    ----------
    - h5_file:
        Path to the h5py file (numpy arrays of the audios).
    - utterances: str, Path
        Path to the file storing the utterances keys.
    - targets: str, Path.
        Path to the file containing targets (entropies) for each utterance.
    - checkpoint: str
        The path to the huggingface checkpoint of the processor.
    - sampling_rate: int
        The sampling rate of the audios. Default=16000
    - sub_hours: Optional
        The number of sub hours to consider. Default=None, meaning\
        all the utterances are considered.
    """

    def __init__(self,
                 h5_file: DataPath,
                 utterances: DataPath,
                 targets: DataPath,
                 checkpoint: str,
                 sampling_rate: int=16000,
                 sub_hours:Optional[int]=None):
        self.targets_path = targets
        self.load_targets()
        self.h5_file = h5py.File(h5_file)
        self.sampling_rate = sampling_rate
        with open(utterances, "r") as sorted_utterances:
            self.ids = [line.strip() for line in sorted_utterances]
        if sub_hours is not None:
            self.subset(sub_hours)
        self.sample_size = len(self.ids)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.has_timemarks = None
    
    def subset(self, sub_hours: int):
        """Subsetting the whole corpus for a given number of hours."""
        LOGGER.info(f"Getting a subset of {sub_hours} hours...")
        finish_message = "Data subsetting finished! Exactly {} hours (for {} utterances) were sampled."
        new_ids = []
        total_hours = 0.0
        pbar = tqdm(total=sub_hours)
        shuffle(self.ids)
        for audio_id in self.ids:
            frames = self.h5_file[audio_id].shape[0]
            hour = (frames / self.sampling_rate) / 3600
            pbar.update(hour)
            total_hours += hour
            new_ids.append((frames, audio_id))
            if total_hours >= sub_hours:
                new_ids = sorted(new_ids)
                _, new_ids = list(zip(*new_ids))
                self.ids = new_ids
                LOGGER.info(finish_message.format(total_hours, len(new_ids)))
                break

    def load_targets(self):
        """Reads and stores the targets."""
        self.utterance_targets = dict()
        with open(self.targets_path, "r") as utterances_targets:
            for target in utterances_targets:
                target.strip()
                utterance_id, entropy = target.split("\t")
                self.utterance_targets[utterance_id] = float(entropy)

    def utterances_iterator(self):
        """An iterator over audio frames."""
        for utterance_id in self.ids:
            yield utterance_id, self.h5_file[utterance_id][:]

    def data_iterator(self) -> Iterator[DataItem]:
        """An iterator over utterances and their corresponding labels."""
        for utterance_id, utterance in self.utterances_iterator():
            entropy = self.utterance_targets[utterance_id]
            yield utterance_id, utterance, entropy

    def __call__(self,
                 batch_size: int=32
                 ) -> Iterator[Tuple[Array, Array, List[UtteranceId]]]:
        """Creates an iterator over batches."""
        iterator = self.data_iterator()
        for first in iterator:
            utterance_ids, utterances, entropies = zip(*chain([first], islice(iterator, batch_size - 1)))
            inputs = self.processor(utterances,
                                    sampling_rate=self.sampling_rate,
                                    return_tensors="pt")
            y = torch.tensor(entropies)
            x = inputs["input_features"]
            yield x, y, list(utterance_ids)