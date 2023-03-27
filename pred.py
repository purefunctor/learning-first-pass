import matplotlib.pyplot as plt
from sphere_world import random_scene_render
import torch
from main import Raycaster

checkpoint = torch.load("training_state.pth")
device = "mps"

raycaster = Raycaster().to(device)
raycaster.load_state_dict(checkpoint["model_state"])
raycaster.eval()

features, image = random_scene_render(1_069_420)
features, image = (
    torch.tensor(features, device=device, dtype=torch.float),
    torch.tensor(image, device=device, dtype=torch.float),
)

print(f"Got features of shape {features.shape}")
print(f"Got target of shape {image.shape}")

features = features.unsqueeze(0)
prediction = raycaster(features).squeeze(0)

print(f"Got prediction of shape {prediction.shape}")

figure = plt.figure()

ax = figure.add_subplot(1, 2, 1)

ax.imshow(image.cpu().detach().numpy().transpose((1, 2, 0)))
ax.invert_yaxis()
ax.set_title("Scene (Target)")

ax = figure.add_subplot(1, 2, 2)

ax.imshow(prediction.cpu().detach().numpy().transpose((1, 2, 0)))
ax.invert_yaxis()
ax.set_title("Scene (Prediction)")

plt.show()
