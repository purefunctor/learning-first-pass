from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sphere_world import SphereWorld

world = SphereWorld(seed=0, angles=9, size=128)

figure = plt.figure()

grid = ImageGrid(figure, 111, nrows_ncols=(3, 3), axes_pad=0.1)

ax = grid[0]

_, image = world.render(angle=0)
ax.imshow(image.transpose((1, 2, 0)))
ax.invert_yaxis()

plt.show()
