"""A basic trainer for the entropy predictor model."""
from data_loader import DataLoader
from model import EntropyWhisper
from pathlib import Path
from argparse import ArgumentParser
import random
import logging
import torch
from tqdm import tqdm
import yaml

torch.manual_seed(1797)
random.seed(1797)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def train(model: EntropyWhisper,
          device: torch.device,
          output_path: Path,
          data_loader: DataLoader,
          model_name: str="model",
          batch_size: int=32,
          epochs=5,
          lr: int=0.00056) -> None:
    """Train the model to predict text entropies from spoken utterances."""
    output_path.mkdir(exist_ok=True, parents=True)
    mse = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("Inf")
    total = 0
    logs = []
    for epoch in range(1, epochs + 1):
        data_iterator = data_loader(batch_size)
        bar = tqdm(total=data_loader.sample_size)
        epoch_losses = 0
        for x, y, _ in data_iterator:
            model.zero_grad()
            x = x.to(device)
            y = y.to(device)
            predicted_entropy = model(x)
            loss = mse(predicted_entropy.squeeze(-1), y)
            loss.backward()
            optimizer.step()

            epoch_losses += loss.item()
            total += 1
            bar.update(x.shape[0])

        epoch_loss = epoch_losses / total
        log = f"epoch={epoch}, train loss={epoch_loss}, lr={optimizer.param_groups[0]['lr']}"
        LOGGER.info(log)
        logs.append(str(epoch_loss))
        if epoch_loss < best_loss :
            best_loss = epoch_loss
            torch.save(model.state_dict(), output_path / f"{model_name}.pt")
    with open(f"{model_name}.epochs", "w") as log_file:
        log_file.write("\n".join(logs))

def main():
    parser = ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        required=True,
                        help="The yaml config (the same used during the training).")
    
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device {device}")
    model = EntropyWhisper(config["checkpoint"]).to(device)
    output_folder = Path(config["output_folder"])
    output_folder.mkdir(exist_ok=True, parents=True)
    data_loader = DataLoader(h5_file=config["h5_data"],
                             utterances=config["utterances"],
                             targets=config["targets"],
                             checkpoint=config["checkpoint"],
                             sub_hours=config["sub_hours"])
    train(model=model,
          device=device,
          output_path=output_folder,
          data_loader=data_loader,
          model_name=config["model_name"],
          batch_size=config["batch_size"],
          epochs=config["epochs"],
          lr=config["learning_rate"])

if __name__ == "__main__":
    main()

    