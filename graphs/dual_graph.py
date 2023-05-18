import logging
import numpy as np
from sklearn.neighbors import kneighbors_graph
import scipy.sparse


def csr_row_set_nz_to_val(csr, row, value=0):
    """
    Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row + 1]] = value
    return csr


def csr_rows_set_nz_to_val(csr, rows, value=0):
    for row in rows:
        csr = csr_row_set_nz_to_val(csr, row)
    if value == 0:
        csr.eliminate_zeros()
    return csr


def dual_graph(image, mask, k=10):
    logging.info(f"Building graph from two partially-overlapping images. Using k={k} nearest neighbours.")

    if not image.ndim == 4:
        msg = f"Expected 4-dimensional image, got {image.ndim} dimensions."
        logging.error(msg)
        raise ValueError(msg)

    print("Building graph for first image...")
    # graph using observed parts of first image
    one_a = kneighbors_graph(image[..., 0].reshape(-1, image.shape[2]), k, include_self=False)
    one_a = one_a + one_a.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
    one_a[one_a > 1] = 1  # get rid of any edges we just made double
    one_a = csr_rows_set_nz_to_val(one_a, np.argwhere(mask[..., 0, 0].flatten() == 0).flatten())
    one_a = scipy.sparse.csr_matrix(csr_rows_set_nz_to_val(scipy.sparse.csr_matrix(one_a.T), np.argwhere(
        mask[..., 0, 0].flatten() == 0).flatten()).T)

    print("Building graph for second image...")
    # graph using observed parts of second image
    two_a = kneighbors_graph(image[..., 1].reshape(-1, image.shape[2]), k, include_self=False)
    two_a = two_a + two_a.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
    two_a[two_a > 1] = 1  # get rid of any edges we just made double
    two_a = csr_rows_set_nz_to_val(two_a, np.argwhere(mask[..., 0, 1].flatten() == 0).flatten())
    two_a = scipy.sparse.csr_matrix(csr_rows_set_nz_to_val(scipy.sparse.csr_matrix(two_a.T), np.argwhere(
        mask[..., 0, 1].flatten() == 0).flatten()).T)

    # combine graphs
    adj = one_a + two_a
    adj[adj > 1] = 1  # get rid of any edges we just made double
    return adj
