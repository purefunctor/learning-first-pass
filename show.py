import matplotlib.pyplot as plt
from sphere_world import random_scene_render

figure = plt.figure()

features, image = random_scene_render()

print(f"Got input features of shape {features.shape}")
print(f"Got expected output of shape {image.shape}")

ax = figure.add_subplot(1, 1, 1)
ax.imshow(image.transpose((1, 2, 0)))
ax.invert_yaxis()
ax.set_title("Scene (Render)")

plt.show()
