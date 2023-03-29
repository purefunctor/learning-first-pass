import math
from pathlib import Path
from matplotlib import pyplot as plt
import torch

from lru import LRU
from sphere_world import SphereWorld
from torch import nn
from torch.utils.data import Dataset, DataLoader


TRAIN_COUNT = 80
TEST_COUNT = 20
TOTAL_COUNT = TRAIN_COUNT + TEST_COUNT

IMAGE_SIZE = 32
BATCH_SIZE = 5
ANGLE_COUNT = 5
WORLD_CACHE = LRU(1)

device = "mps" if torch.backends.mps.is_available() else "cpu"


class Images(Dataset):
    def __init__(self, *, epoch=0, testing=False):
        self.epoch = epoch
        self.testing = testing

    def __getitem__(self, index):
        index += TOTAL_COUNT * self.epoch

        if self.testing:
            index += TRAIN_COUNT

        index, angle = divmod(index, ANGLE_COUNT)

        if index not in WORLD_CACHE:
            WORLD_CACHE[index] = SphereWorld(
                seed=index, angles=ANGLE_COUNT, size=IMAGE_SIZE
            )

        features, target = WORLD_CACHE[index].render(angle=angle)

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

        self.conv3d_0 = nn.Conv3d(
            in_channels=19, out_channels=64, kernel_size=5, padding="same"
        )
        self.batchNorm3d_0 = nn.BatchNorm3d(64)
        self.maxpool_0 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.relu_0 = nn.ReLU()

        self.conv3d_1 = nn.Conv3d(
            in_channels=64, out_channels=32, kernel_size=5, padding="same"
        )
        self.batchNorm3d_1 = nn.BatchNorm3d(32)
        self.maxpool_1 = nn.MaxPool3d(
            kernel_size=(16, 1, 1), stride=(1, 1, 1)
        )
        self.relu_1 = nn.ReLU()

        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding="same")
        self.batchNorm2d_2 = nn.BatchNorm2d(16)
        self.relu_2 = nn.ReLU()

        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding="same")
        self.batchNorm2d_3 = nn.BatchNorm2d(3)
        self.relu_3 = nn.ReLU()

        self.sigmoid_4 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv3d_0(x)
        x = self.batchNorm3d_0(x)
        x = self.maxpool_0(x)
        x = self.relu_0(x)

        x = self.conv3d_1(x)
        x = self.batchNorm3d_1(x)
        x = self.maxpool_1(x)
        x = self.relu_1(x)

        x = torch.sum(x, dim=2)

        x = self.conv2d_2(x)
        x = self.batchNorm2d_2(x)
        x = self.relu_2(x)

        x = self.conv2d_3(x)
        x = self.batchNorm2d_3(x)
        x = self.relu_3(x)

        return self.sigmoid_4(x)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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


if __name__ == "__main__":
    device = "cpu"

    print(f"Using Device: {device}")

    raycaster = Raycaster().to(device)
    loss_fn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(raycaster.parameters(), learning_rate)

    if not Path("training_state.pth").exists():
        torch.save(
            {
                "epochs": 0,
                "model_state": raycaster.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            "training_state.pth",
        )

    checkpoint = torch.load("training_state.pth")
    epochs = checkpoint["epochs"]

    raycaster.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    for epoch in range(epochs, epochs + 5):
        print(f"Epoch {epoch}")

        training_data = Images(epoch=epoch, testing=False)
        test_data = Images(epoch=epoch, testing=True)

        training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

        raycaster.train()
        train(training_dataloader, raycaster, loss_fn, optimizer)

        raycaster.eval()
        test(test_dataloader, raycaster, loss_fn)

    print("Done!")

    torch.save(
        {
            "epochs": epochs + 5,
            "model_state": raycaster.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        "training_state.pth",
    )
