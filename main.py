import json
from pathlib import Path

from torch.utils.data import Dataset


def image_file_index(image_file: Path):
    return int(str(image_file.name).removeprefix("image_").removesuffix(".json"))


class Images(Dataset):
    def __init__(self, image_directory: Path = None):
        if image_directory is None:
            image_directory = Path.cwd() / "output"
        self.image_directory = image_directory
        self.image_files = list(image_directory.glob("*.json"))
        self.image_files.sort(key=image_file_index)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        with image_file.open("r") as f:
            return image_file.name, json.load(f)


training_images = Images()
