import numpy as np


def reverse_never_observed(features, reversing_mask):
    if not features.ndim == 2:
        raise ValueError("Expected 2-dimensional features. Input should be (#nodes, #features)")
    output = np.zeros((reversing_mask.shape[0], features.shape[1]), dtype=features.dtype)
    output[reversing_mask] = features
    return output
