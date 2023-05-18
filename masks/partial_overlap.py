import numpy as np


def partial_overlap(shape, day_0_remove_n_rows=90, day_0_remove_n_cols=90, day_1_remove_n_rows=90,
                    day_1_remove_n_cols=90):
    """
    Creates a boolean mask replicating if two acquisitions partially overlap

    returns a 4th-order tensor (rows, cols, channels, temporal)
    """
    mask = np.ones(shape + (2,), dtype=np.bool)

    mask[:day_0_remove_n_rows, ..., 0] = False
    mask[:, :day_0_remove_n_cols, ..., 0] = False
    mask[-day_1_remove_n_rows:, ..., 1] = False
    mask[:, -day_1_remove_n_cols:, ..., 1] = False

    return mask
