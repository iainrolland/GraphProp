import numpy as np


def slc_off_mask(shape):
    """
    Creates a boolean mask replicating what might be observed in a Landsat 7 image after the Scan Line Corrector failure
    """
    x_pixels = np.tile(np.arange(shape[1])[None, :], (shape[0], 1)) + 1e-6
    y_pixels = np.tile(np.arange(shape[0])[:, None], (1, shape[1])) + 1e-6
    slope_dist = np.cos(np.arctan(y_pixels / x_pixels) + (np.pi / 2 - np.arctan(0.179))) * (
            x_pixels ** 2 + y_pixels ** 2) ** 0.5
    mask = ((slope_dist % 31.4) / 31.4) > (4 / 9.5)
    if len(shape) > 2:
        mask = np.stack([mask] * shape[2], axis=-1)
    return mask
