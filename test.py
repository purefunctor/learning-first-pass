import torch

from main import Raycaster, Images
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

device = "mps"
raycaster = Raycaster().to(device)
checkpoint = torch.load("training_state.pth")
raycaster.load_state_dict(checkpoint["model_state"])
raycaster.eval()

test_images = Images(image_directory="output-train")
test_images_loader = DataLoader(test_images)

X, y = next(iter(test_images_loader))

X, y = X.to(device), y.to(device)

y_hat = raycaster(X)

y_image = ToPILImage()(y[0])
y_hat_image = ToPILImage()(y_hat[0])

y_image.show()
y_hat_image.show()
