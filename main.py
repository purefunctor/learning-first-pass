from argparse import ArgumentParser
import math
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import wandb

from lru import LRU
from sphere_world import SphereWorld
from torch import nn
from torch.utils.data import Dataset, DataLoader


TRAIN_COUNT = 1000
TEST_COUNT = 500
TOTAL_COUNT = TRAIN_COUNT + TEST_COUNT

IMAGE_SIZE = 64
BATCH_SIZE = 100

device = "mps" if torch.backends.mps.is_available() else "cpu"


class Images(Dataset):
    def __init__(self, *, seed=0, testing=False, angle_count=10, vertical_count=10):
        self.seed = seed
        self.testing = testing
        self.angle_count = angle_count
        self.vertical_count = vertical_count
        self.world_cache = LRU(
            math.ceil(BATCH_SIZE / (self.angle_count * self.vertical_count))
        )

    def __getitem__(self, index):
        index += TOTAL_COUNT * self.seed

        if self.testing:
            index += TRAIN_COUNT

        index, angle = divmod(index, self.angle_count * self.vertical_count)
        angle, vertical = divmod(angle, self.angle_count)

        if index not in self.world_cache:
            self.world_cache[index] = SphereWorld(
                seed=index,
                angles=self.angle_count,
                verticals=self.vertical_count,
                size=IMAGE_SIZE,
            )

        features, target = self.world_cache[index].render(
            angle=angle, vertical=vertical
        )

        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(target, dtype=torch.float),
        )

    def __len__(self):
        if self.testing:
            return TEST_COUNT
        else:
            return TRAIN_COUNT

class Raycaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=23, out_channels=128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding="same"),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequence(x)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        total_loss += loss
    return total_loss / len(dataloader)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            current = (batch + 1) * len(X)
            print(f"Total Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    loss /= len(dataloader)
    print(f"Test Error: \n Average Loss: {loss:>7f} \n")
    return loss


if __name__ == "__main__":
    parser = ArgumentParser(prog="learning-first-pass")

    parser.add_argument("--epochs")
    parser.add_argument("--delta")
    parser.add_argument("--mode")
    parser.add_argument("--seed")
    parser.add_argument("--train-angle")
    parser.add_argument("--train-vertical")
    parser.add_argument("--test-angle")
    parser.add_argument("--test-vertical")

    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using Device: {device}")

    raycaster = Raycaster().to(device)
    loss_fn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(raycaster.parameters(), learning_rate)

    if not Path("training_state.pth").exists():
        torch.save(
            {
                "model_state": raycaster.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            "training_state.pth",
        )

    checkpoint = torch.load("training_state.pth")
    raycaster.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    epochs = int(args.epochs)
    epoch_delta = int(args.delta)
    train_angle = int(args.train_angle)
    train_vertical = int(args.train_vertical)
    test_angle = int(args.test_angle)
    test_vertical = int(args.test_vertical)
    seed = int(args.seed)

    wandb.init(
        project="learning-first-pass",
        config={
            "mode": args.mode,
            "seed": seed,
            "train_angle": train_angle,
            "train_vertical": test_vertical,
            "test_angle": test_angle,
            "test_vertical": test_vertical,
        },
    )

    for epoch in range(epochs, epochs + epoch_delta):
        print(f"Epoch {epoch}, Seed: {seed}")

        training_data = Images(
            seed=seed,
            testing=False,
            angle_count=train_angle,
            vertical_count=train_vertical,
        )
        test_data = Images(
            seed=seed,
            testing=True,
            angle_count=test_angle,
            vertical_count=test_vertical,
        )

        training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

        if args.mode == "train":
            raycaster.train()
            training_loss = train(training_dataloader, raycaster, loss_fn, optimizer)
            wandb.log({"training_loss": training_loss}, step=epoch)

        raycaster.eval()
        testing_loss = test(test_dataloader, raycaster, loss_fn)
        wandb.log({"testing_loss": testing_loss}, step=epoch)

        seed += 1

    print("Done!")

    torch.save(
        {
            "model_state": raycaster.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        "training_state.pth",
    )

    wandb.finish()
