from PIL import Image
import numpy as np


def read_image_from_file(file, image_size=(512, 512)):
    image = Image.open(file).convert('L')
    image = image.resize(image_size)
    image = np.array(image).astype('float32')
    image = image / 255.0
    return image 


def random_small_rotation(scale = 0.1):
    angle = np.random.normal(0, scale = scale)
    matrix = np.array([
        [np.cos(angle), np.sin(angle)],
        [-1.0 * np.sin(angle), np.cos(angle)]
    ])
    return matrix





