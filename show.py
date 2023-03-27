from matplotlib import pyplot as plt
from sphere_world import SphereWorld

world = SphereWorld(seed=0, angles=4, size=1024)

figure = plt.figure()

features0, image0 = world.render(angle=0)
features1, image1 = world.render(angle=2)

print(f"Got features of shape {features0.shape}, {features1.shape}")
print(f"Got target of shape {image0.shape}, {image1.shape}")

ax = figure.add_subplot(1, 2, 1)

ax.imshow(image0.transpose((1, 2, 0)))
ax.invert_yaxis()
ax.set_title("Scene (Render)")

ax = figure.add_subplot(1, 2, 2)

ax.imshow(image1.transpose((1, 2, 0)))
ax.invert_yaxis()
ax.set_title("Scene (Render)")

plt.show()
