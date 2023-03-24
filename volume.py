import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

with Path("output-test/render_0.json").open() as f:
    render_data = np.array(json.load(f))

with Path("output-test/volume_0.json").open() as f:
    volume_data = np.array(json.load(f))

figure = plt.figure()

ax = figure.add_subplot(1, 2, 1)
ax.imshow(render_data.transpose())
ax.invert_yaxis()
ax.set_title("Scene (Render)")

ax = figure.add_subplot(1, 2, 2, projection="3d")
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_title("Scene (Volume)")
volume_data = volume_data.transpose((3, 1, 2, 0))

r, g, b = np.indices((volume_data.shape[0] + 1, volume_data.shape[0] + 1, volume_data.shape[0] + 1))
sphere = volume_data[..., 0:2].any(3)
colors = np.zeros(sphere.shape + (3,))
colors[..., 0] = volume_data[..., 3]
colors[..., 1] = volume_data[..., 4]
colors[..., 2] = volume_data[..., 5]
ax.voxels(r, g, b, sphere, facecolors=colors)

plt.show()
