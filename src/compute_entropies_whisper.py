"""Use the model trained on top of Whisper features\
    for computing entropy on spoken utterances."""

from model import EntropyWhisper
from data_loader import DataLoader
from typing import List, Optional
import logging
from pathlib import Path
from math import exp
from argparse import ArgumentParser
from pandas import DataFrame
from tqdm import tqdm
import torch
import yaml

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def compute_entropies(model: EntropyWhisper,
                      utterances: torch.Tensor
                      ) ->  List[float]:
    """Computes the entropies of a batch of spoken utterances\
        using the trained model."""
    with torch.no_grad():
        entropies = model(utterances)
    return entropies.tolist()

def compute_metrics(whisper_checkpoint: str,
                    model_checkpoint: str,
                    data_loader: DataLoader,
                    batch_size: Optional[int]=32
                    ) -> DataFrame:
    """Computes entropies on all data and save them into a DataFrame."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using {device}...")
    model = EntropyWhisper(whisper_checkpoint)
    state_dict = torch.load(model_checkpoint, map_location=device)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    results = []
    bar = tqdm(total=data_loader.sample_size)
    for batch, gold_entropies, utterances_ids in data_loader(batch_size):
        batch = batch.to(device)
        entropies = compute_entropies(model, batch)
        all_informations = zip(entropies, gold_entropies.tolist(), utterances_ids)
        for entropy, gold_entropy, utterance_id in all_informations:
            results.append({
                "entropy": entropy,
                "perplexity": exp(entropy),
                "gold_entropy": gold_entropy,
                "utterance_id": utterance_id})
        bar.update(batch.shape[0])
    return DataFrame(results)

def main():
    parser = ArgumentParser()
    parser.add_argument("-c",
                    "--config",
                    required=True,
                    help="The yaml config (the same used during the training).")
    
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    output_folder = Path("results")
    output_folder.mkdir(exist_ok=True, parents=True)
    data_loader = DataLoader(h5_file=config["h5_data"],
                             utterances=config["utterances"],
                             targets=config["targets"],
                             checkpoint=config["checkpoint"])
    
    results_df = compute_metrics(whisper_checkpoint=config["checkpoint"],
                                 model_checkpoint=config["model_checkpoint"],
                                 data_loader=data_loader,
                                 batch_size=config["batch_size"])
    results_df.to_csv(output_folder / f"{config['output_filename']}.csv")

if __name__ == "__main__":
    main()