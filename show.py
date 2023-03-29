from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from sphere_world import SphereWorld

world = SphereWorld(seed=0, angles=9, size=32)

volume, image = world.render(angle=0)
volume, image = (
    volume.transpose((3, 1, 2, 0)),
    image.transpose((1, 2, 0)),
)

figure = plt.figure()

ax = figure.add_subplot(1, 2, 1)

ax.imshow(image)
ax.invert_yaxis()

ax = figure.add_subplot(1, 2, 2, projection="3d")

r, g, b = np.indices(
    (volume.shape[0] + 1, volume.shape[0] + 1, volume.shape[0] + 1)
)
sphere = volume[..., 0:2].any(3)
colors = np.zeros(sphere.shape + (3,))
colors[..., 0] = volume[..., 4]
colors[..., 1] = volume[..., 5]
colors[..., 2] = volume[..., 6]

ax.voxels(r, g, b, sphere, facecolors=colors)
ax.invert_yaxis()

plt.show()
