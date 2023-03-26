from pathlib import Path
import torch

from sphere_world import random_scene_render
from torch import nn
from torch.utils.data import Dataset, DataLoader

IMAGE_COUNT = 10_000
TEST_THRESHOLD = 0.2

class Images(Dataset):
    def __init__(self, *, testing = False):
        self.testing = testing

    def __getitem__(self, index):
        if self.testing:
            index += int(IMAGE_COUNT * (1 - TEST_THRESHOLD))
        features, target = random_scene_render(index)
        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(target, dtype=torch.float),
        )

    def __len__(self):
        if self.testing:
            return int(IMAGE_COUNT * TEST_THRESHOLD)
        else:
            return int(IMAGE_COUNT * (1 - TEST_THRESHOLD))


class Raycaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding="same"),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequence(x)


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
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    loss /= len(dataloader)
    print(f"Test Error: \n Average Loss: {loss:>7f} \n")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using Device: {device}")

    training_data = Images(testing=False)
    test_data = Images(testing=True)

    training_dataloader = DataLoader(training_data, batch_size=100)
    test_dataloader = DataLoader(test_data, batch_size=100)

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

    for epoch in range(epochs, epochs + 10):
        print(f"Epoch {epoch}")

        raycaster.train()
        train(training_dataloader, raycaster, loss_fn, optimizer)

        raycaster.eval()
        test(test_dataloader, raycaster, loss_fn)

    print("Done!")

    torch.save(
        {
            "epochs": epochs + 10,
            "model_state": raycaster.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        "training_state.pth",
    )
