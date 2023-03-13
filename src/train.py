from data_loader import DataLoader
from pathlib import Path
from argparse import ArgumentParser
from model import EntropyWhisper
import torch
import random
from tqdm import tqdm
import logging

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
            # reinitialize gradients to zeros
            model.zero_grad()
            x = x.to(device)
            y = y.to(device)
            # predict the entropy of the speech utterance
            predicted_entropy = model(x)
            # compute the mean squared error of the predicted and target entropies
            loss = mse(predicted_entropy.squeeze(-1), y)
            # compute the derivatives of the loss function in respect to the models parameters
            loss.backward()
            # update the parameters of the model using the gradient values
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
    parser = ArgumentParser(description='A script for training text entropy prediction from a give.')
    parser.add_argument("-c", "--checkpoint",
                        required=True,
                        help="Huggingface path to the whisper model.")
    parser.add_argument("-o", "--output_folder",
                        required=True,
                        help="Where the trained model will be stored.")
    parser.add_argument("-b", "--batch_size",
                        default=64,
                        type=int,
                        help="The batch size.")
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=5,
                        help="The number of epochs")
    parser.add_argument("-l", "--learning_rate",
                        type=float,
                        default=0.00056,
                        help="The number of epochs")
    parser.add_argument("-u", "--sorted_utterances",
                        required=True,
                        help="File containing ids sorted by utterances length.")
    parser.add_argument("-d", "--h5_data",
                        required=True,
                        help="Path to the file containing utterances.")
    parser.add_argument("-t", "--targets",
                        required=True,
                        help="Path to the targets for each utterance in the utterances_file.")
    parser.add_argument("-s", "--sub_hours",
                        required=False,
                        default=None,
                        type=int,
                        help="The number of hours to use for training.")
    parser.add_argument("-n", "--model_name",
                        required=True,
                        help="The name of the model.")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device {device}")
    model = EntropyWhisper(args.checkpoint).to(device)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    data_loader = DataLoader(args.h5_data,
                             args.sorted_utterances,
                             args.targets,
                             args.checkpoint,
                             sub_hours=args.sub_hours)
    train(model, device, output_folder, data_loader, args.model_name, args.batch_size, args.epochs, args.learning_rate)

if __name__ == "__main__":
    main()

    