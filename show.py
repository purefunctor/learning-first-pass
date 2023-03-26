import matplotlib.pyplot as plt
from sphere_world import random_scene_render

figure = plt.figure()

render = random_scene_render()

ax = figure.add_subplot(1, 2, 1)
ax.imshow(render)
ax.invert_yaxis()
ax.set_title("Scene (Render)")

plt.show()
