import os
import glob

import numpy as np
from keras.preprocessing import image as k_image

from .utils import read_image_from_file, random_small_rotation
from .gauss import add_gaussian_on_image


class ImageGenerator(object):
    
    _PATH  = './photo'
    
    def __init__(self, path=_PATH, image_size=(512, 512), 
                 dark_min=50, dark_max=150, 
                 rad_min=0.0001, rad_max=0.0005, 
                 interpoint_dist_mean=0.1, interpoint_dist_std=0.01):
        
        self.files = sorted(glob.glob(os.path.join(path, '*.bmp')))
        self.image_size = image_size
        self.dark_min = dark_min
        self.dark_max = dark_max
        self.rad_min = rad_min
        self.rad_max = rad_max
        self.interpoint_dist_mean = interpoint_dist_mean
        self.interpoint_dist_std = interpoint_dist_std
              
    def __get_random_background(self):
        random_path = np.random.choice(self.files, 1)
        array = read_image_from_file(random_path[0], self.image_size)
        array = array.reshape(self.image_size + (1,)) # add extra dim
        array = k_image.random_rotation(array, 90.0, row_axis=0, 
                                        col_axis=1, channel_axis=2, fill_mode='reflect')
        
        array = k_image.random_shift(array, 0.1, 0.1, row_axis=0, 
                                     col_axis=1, channel_axis=2, fill_mode='reflect')
        return array[:, :, 0]
    
    def get_images(self, n_images=3): 
        # Background
        dark_color = np.random.randint(self.dark_min, self.dark_max)
        light_color = np.random.randint(1, 255 - dark_color)

        img = self.__get_random_background() * dark_color
        t_vec = np.random.random(size=2)
        t_vec = t_vec / np.linalg.norm(t_vec)

        images = []
        masks = []

        point = np.random.uniform(-0.04, 1.04, 2)
        
        radius = np.random.uniform(self.rad_min, self.rad_max)
        cov = np.eye(2) #+ 0.5 * np.random.uniform(-1, 1, size=(2, 2))

        for _ in range(n_images):
            generated_img = add_gaussian_on_image(img, light_color, point, radius, cov)
            images.append(generated_img)

            mask = add_gaussian_on_image(np.zeros_like(img), 1.0, point, radius, cov) > 0.5
            mask = mask.astype('float32')
            masks.append(mask)

            t_vec = random_small_rotation().dot(t_vec)
            distance = np.random.normal(self.interpoint_dist_mean, self.interpoint_dist_std)
            point += t_vec * distance
            # cov = random_small_rotation().dot(cov)

        return np.stack(images, axis=-1), np.stack(masks, axis=-1)

