import json
from os import PathLike
from pathlib import Path
import re
import torch

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


def image_file_index(image_file: Path):
    return int(str(image_file.name).removeprefix("image_").removesuffix(".json"))


FILE_INDEX = re.compile("(render|volume)_(\d+).json")


def get_file_index(image_file: Path):
    return int(FILE_INDEX.search(str(image_file.name)).group(2))


class Images(Dataset):
    def __init__(self, image_directory: PathLike = None):
        if image_directory is None:
            image_directory = Path.cwd() / "output"
        else:
            image_directory = Path(image_directory)
        self.image_directory = image_directory

        self.render_files = list(image_directory.glob("render_*.json"))
        self.volume_files = list(image_directory.glob("volume_*.json"))

        self.render_files.sort(key=get_file_index)
        self.volume_files.sort(key=get_file_index)

        assert len(self.render_files) == len(self.volume_files)

    def __getitem__(self, index):
        render_json = self.render_files[index]
        volume_json = self.volume_files[index]
        with render_json.open("r") as f_r, volume_json.open("r") as f_v:
            return (
                torch.tensor(json.load(f_v), dtype=torch.float),
                torch.tensor(json.load(f_r), dtype=torch.uint8),
            )

    def __len__(self):
        return len(self.render_files)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv3d_0 = nn.Conv3d(
            in_channels=7, out_channels=128, kernel_size=5, padding="same"
        )
        self.batchNorm3d_0 = nn.BatchNorm3d(128)
        self.maxpool_0 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )

        self.conv3d_1 = nn.Conv3d(
            in_channels=128, out_channels=64, kernel_size=5, padding="same"
        )
        self.batchNorm3d_1 = nn.BatchNorm3d(64)
        self.maxpool_1 = nn.MaxPool3d(
            kernel_size=(2, 1, 1), stride=(2, 1, 1)
        )

    def forward(self, x):
        x = self.conv3d_0(x)
        x = self.batchNorm3d_0(x)
        x = self.maxpool_0(x)

        x = self.conv3d_1(x)
        x = self.batchNorm3d_1(x)
        x = self.maxpool_1(x)

        # Return the output tensor
        return x


device = "cpu"

print(f"Using Device: {device}")


def train(dataloader, model, loss_fn, optimizer):
    model.train()
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
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        raycaster.eval()
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Average Loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    training_data = Images(image_directory="output-train")
    test_data = Images(image_directory="output-test")

    training_dataloader = DataLoader(training_data, batch_size=2)
    test_dataloader = DataLoader(test_data, batch_size=2)

    raycaster = Model().to(device)
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
        train(training_dataloader, raycaster, loss_fn, optimizer)
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
