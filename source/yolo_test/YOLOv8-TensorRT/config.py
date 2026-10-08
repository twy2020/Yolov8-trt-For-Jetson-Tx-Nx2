import random
import numpy as np

random.seed(0)

# detection model classes
CLASSES_DET = (
    'red_ring',
    'green_ring',
    'blue_ring',
)

# colors for each detection class
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for cls in CLASSES_DET
}
