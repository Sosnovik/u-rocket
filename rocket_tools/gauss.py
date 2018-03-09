import numpy as np


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


def add_gaussian_on_image(image, magnitude, mean, radius, cov):        
    new_image = image.copy()
    gauss = gaussian2D(image.shape, mean, radius * cov)
    new_image += gauss * magnitude
    return new_image
    
    
    
    