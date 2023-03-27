import matplotlib.pyplot as plt
from sphere_world import random_scene_render

figure = plt.figure()

features0, image0 = random_scene_render(61_300)

print(f"Got features of shape {features0.shape}")
print(f"Got target of shape {image0.shape}")

ax = figure.add_subplot(1, 1, 1)

ax.imshow(image0.transpose((1, 2, 0)))
ax.invert_yaxis()
ax.set_title("Scene (Render)")

plt.show()
