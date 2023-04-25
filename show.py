import numpy as np

from cpu_tracer import render
from PIL import Image

image ,= render(800, 600, 90)
image = Image.fromarray(np.uint8(255 * image), "RGB")
image.show()
