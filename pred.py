import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sphere_world import SphereWorld
import torch
from main import Raycaster
import time

checkpoint = torch.load("training_state.pth")
device = "mps"
raycaster = Raycaster().to("mps")
raycaster.load_state_dict(checkpoint["model_state"])
raycaster.eval()

world_0 = SphereWorld(seed=1_069_420, angles=10, verticals=10, size=128)

figure = plt.figure()

grid = ImageGrid(figure, 111, nrows_ncols=(2, 2), axes_pad=0.1)

features_0, image_0 = world_0.render(angle=1, vertical=1)
features_1, image_1 = world_0.render_random()

features_0, image_0 = (
    torch.tensor(features_0, device="mps", dtype=torch.float),
    torch.tensor(image_0, device="mps", dtype=torch.float),
)

features_1, image_1 = (
    torch.tensor(features_1, device="mps", dtype=torch.float),
    torch.tensor(image_1, device="mps", dtype=torch.float),
)

features_0 = features_0.unsqueeze(0)
features_1 = features_1.unsqueeze(0)

# TODO: benchmark this...
prediction_0 = raycaster(features_0).squeeze(0)
prediction_1 = raycaster(features_1).squeeze(0)

ax = grid[0]

ax.imshow(image_0.cpu().detach().numpy().transpose((1, 2, 0)))
ax.invert_yaxis()

ax = grid[1]

ax.imshow(prediction_0.cpu().detach().numpy().transpose((1, 2, 0)))
ax.invert_yaxis()

ax = grid[2]
ax.imshow(image_1.cpu().detach().numpy().transpose((1, 2, 0)))
ax.invert_yaxis()

ax = grid[3]

ax.imshow(prediction_1.cpu().detach().numpy().transpose((1, 2, 0)))
ax.invert_yaxis()

plt.show()
