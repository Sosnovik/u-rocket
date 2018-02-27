import numpy as np
import pandas as pd

from keras.preprocessing import image as k_image
import glob
from PIL import Image

image_size = (512, 512)
INTERPOINT_D = 0.1
N_IMAGES = 3

files = glob.glob('photo/*.bmp')
image_size = (512, 512)

def get_array(file):
    image = Image.open(file).convert('L')
    image = image.resize(image_size)
    image = np.array(image).astype('float32')
    image = image / 255.0
    return image 

def get_random_background():
    random_path = np.random.choice(files, 1)
    array = get_array(random_path[0])
    array = array.reshape(image_size + (1,)) # add extra dim
    array = k_image.random_rotation(array, 90.0, row_axis=0, col_axis=1, channel_axis=2, fill_mode='reflect')
    array = k_image.random_shift(array, 0.1, 0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='reflect')
    return array[:, :, 0] # remove extra dim

def gaussian2D(shape, mean, cov):
    '''
    add gaussian to source image
    params:
        shape - shape of the output
        mean - array-like tuple of x_mean, y_mean - relative positions
        cov - 2x2 covariance matrix
        magnitude - maximal intensity of gaussian
    '''
    X, Y = np.indices(shape)
    X = (X + 1).astype(np.float) / (shape[0] + 1)
    Y = (Y + 1).astype(np.float) / (shape[1] + 1)
    r = np.stack([X, Y], -1) - mean
    exp = np.einsum('ijk,kl,ijl->ij', r, np.linalg.inv(cov), r)
    exp = np.exp(-0.5 * exp)
    return exp

def random_small_rotation():
    angle = np.random.normal(0, scale=0.1)
    matrix = np.array([
        [np.cos(angle), np.sin(angle)],
        [-1.0 * np.sin(angle), np.cos(angle)]
    ])
    return matrix

def add_gaussian_on_image(image, magnitude, mean, radius, cov):        
    new_image = image.copy()
    gauss = gaussian2D(image.shape, mean, radius * cov)
    new_image += gauss * magnitude
    return new_image

def gen_random_images():

    # Background
    dark_color = np.random.randint(50, 150)
    light_color = np.random.randint(1, 255 - dark_color)
    
    img = get_random_background() * dark_color
    t_vec = np.random.random(size=2)
    t_vec = t_vec / np.linalg.norm(t_vec)
    
    images = []
    masks = []
    
    
    point = np.random.random(size=2)
    radius = np.random.uniform(0.0001, 0.0025)
    cov = np.eye(2) + 0.5 * np.random.uniform(-1, 1, size=(2, 2))
    
    for _ in range(N_IMAGES):
        generated_img = add_gaussian_on_image(img, light_color, point, radius, cov)
        images.append(generated_img)
        
        mask = add_gaussian_on_image(np.zeros_like(img), 1.0, point, radius, cov) > 0.5
        mask = mask.astype('float32')
        masks.append(mask)
        
        t_vec = random_small_rotation().dot(t_vec)
        distance = 0.1 * np.random.normal(1, scale=0.1)
        point += t_vec * distance
        cov = random_small_rotation().dot(cov)
        radius *= np.random.uniform(0.7, 0.9)
        
    # White noise
    noise = np.random.randint(0, 200, size=image_size)
    p_noise = np.random.random(size=image_size)
    th = np.random.uniform(0, 0.1)
    for img in images:
#         pass
        img[p_noise < th] = noise[p_noise < th]
#         new_images.append(np.expand_dims(img[i], -1
    
    return np.stack(images, axis=-1), np.stack(masks, axis=-1)