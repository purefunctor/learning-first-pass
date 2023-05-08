from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sphere_world import SphereWorld
import time

world = SphereWorld(seed=0, angles=3, verticals=3, size=64)

figure = plt.figure()

grid = ImageGrid(figure, 111, nrows_ncols=(3, 3), axes_pad=0.1)

for i in range(3):
    for j in range(3):
        ax = grid[i * 3 + j]
        start = time.perf_counter()
        _, image = world.render(angle=i, vertical=j)
        print(time.perf_counter() - start)
        ax.imshow(image.transpose((1, 2, 0)))
        ax.invert_yaxis()

plt.show()
