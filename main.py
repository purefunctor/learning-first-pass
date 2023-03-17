import json
from os import PathLike
from pathlib import Path
import torch

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


def image_file_index(image_file: Path):
    return int(str(image_file.name).removeprefix("image_").removesuffix(".json"))


class Images(Dataset):
    def __init__(self, image_directory: PathLike = None):
        if image_directory is None:
            image_directory = Path.cwd() / "output"
        else:
            image_directory = Path(image_directory)
        self.image_directory = image_directory
        self.image_files = list(image_directory.glob("*.json"))
        self.image_files.sort(key=image_file_index)

    def __getitem__(self, index):
        image_json = self.image_files[index]
        image_png = self.image_files[index].with_suffix(".png")
        image = ToTensor()(Image.open(image_png).convert("RGB"))
        with image_json.open("r") as f:
            return torch.tensor(json.load(f), dtype=torch.float), image

    def __len__(self):
        return len(self.image_files)


class Raycaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding="same"),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, padding="same"),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=3, padding="same"),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequence(x)


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using Device: {device}")

training_data = Images(image_directory="output-train")
test_data = Images(image_directory="output-test")

training_dataloader = DataLoader(training_data, batch_size=100)
test_dataloader = DataLoader(test_data, batch_size=100)

raycaster = Raycaster().to(device)
loss_fn = nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(raycaster.parameters(), learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(training_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Average Loss: {test_loss:>8f} \n")

epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    train(training_dataloader, raycaster, loss_fn, optimizer)
    test(test_dataloader, raycaster, loss_fn)
print("Done!")
