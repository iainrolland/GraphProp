import logging
import scipy.sparse as sp
import numpy as np

import common.image_utils as iu
import common.adj_utils as au
from diffusion import _iterative


def complete(adj, gappy_tens, omega, thresh=0.001, iterative=True):
    """
    Given an adjacency matrix, a matrix of entries and a mask (omega) denoting which entries have been observed, use
    graph signal inpainting via total variation minimization and return a completed matrix/tensor.

    - If `gappy_tens' is one-dimensional, we flatten into a vector (i.e. we consider data as matrix with a single band)
    - If `gappy_tens' is two-dimensional, we flatten into a vector (i.e. we consider data as matrix with a single band)
    - If `gappy_tens' is three-dimensional, we flatten the first two dimensions and use the last as the bands

    adj: adjacency matrix (unweighted and undirected), scipy.sparse matrix
    gappy_tens: array, np.ndarray (two-dimensional or three-dimensional array of entries)
    omega: array, np.ndarray (1 if observed, 0 if missing)
    thresh: float (only used if `iterative' is True)
    iterative: bool, if True then we solve using diffusion equations if False we solve for steady state using analytical
    solution (not necessarily always faster - often slower)
    """
    logging.info(f"Completing array of shape {gappy_tens.shape} using GTVM")
    if omega.shape == gappy_tens.shape:
        if omega.ndim == 1:
            pass  # no check necessary, we assume the only dimension is spatial
        elif omega.ndim == 2:
            pass  # no check necessary, we assume both dimensions are spatial
        elif omega.ndim == 3:
            # in this case we assume last dim is spectral
            # we require bands to go missing together (i.e. omega the same for each band)
            # to check, sum in axis=-1 and check that there are only two unique values (or 1 if all missing/observed)
            if not len(np.unique(omega.sum(axis=-1).flatten())) in [1, 2]:
                msg = f"mask with values missing in some but not all bands not yet supported"
                logging.error(msg)
                raise ValueError(msg)
            else:
                omega = omega[..., 0]  # take one band (as we have assumed they are all the same)
        else:
            msg = f"Cannot handle `gappy_tens' of dimension {gappy_tens.ndim}"
            logging.error(msg)
            raise ValueError(msg)
    elif omega.shape != gappy_tens.shape[:2]:
        raise ValueError(
            f"Shape of `omega' {omega.shape} must match first two dimensions of `gappy_tens' {gappy_tens.shape}"
        )

    omega = (omega == 1).flatten()  # flattened bool array
    if gappy_tens.ndim == 1:
        observed = gappy_tens.copy().reshape(-1, 1)
    elif gappy_tens.ndim == 2:
        observed = gappy_tens.copy().reshape(-1, 1)
    elif gappy_tens.ndim == 3:
        observed = gappy_tens.copy().reshape(-1, gappy_tens.shape[2])
    # already checked for dims not in [1, 2, 3]

    adj, observed, reversing_mask = au.never_observed_check(adj, observed, with_reversing_mask=True)
    omega = omega[reversing_mask]

    # compute degree/laplacian matrix from adjacency
    lambda_max, _ = sp.linalg.eigs(adj, k=1, which='LM')  # returns the single largest magnitude eigenvalue
    lambda_max = float(np.abs(np.squeeze(lambda_max)))

    # complete gappy tensor
    logging.info("Normalising adjacency matrix...")
    if iterative:
        normed_adj = adj / np.abs(lambda_max)
        tilde_adj = sp.eye(normed_adj.shape[0]) - normed_adj
        tilde_adj = tilde_adj.T.dot(tilde_adj)
        completed = _iterative(observed,
                               diffuser=(-tilde_adj.astype(np.float32)) / 2,
                               omega=omega,
                               thresh=thresh)
    else:
        completed = _analytical(observed, adj / np.abs(lambda_max), omega)

    # account for any filtered nodes (no edges - i.e. never observed)
    completed = iu.reverse_never_observed(completed, reversing_mask)

    return completed.reshape(gappy_tens.shape)  # return completed tensor in same shape as gappy tensor was provided


def _analytical(observed, normed_adj, omega):
    tilde_adj = sp.eye(normed_adj.shape[0]) - normed_adj
    tilde_adj = tilde_adj.T.dot(tilde_adj)
    completed = np.zeros_like(observed)
    completed[omega == 1] = observed[omega == 1]
    logging.info("Solving analytically...")
    logging.info(tilde_adj[omega == 0][:, omega == 0].shape, observed[omega == 1].shape,
                 -tilde_adj[omega == 0][:, omega == 1].shape)
    completed[omega == 0] = (
        sp.linalg.spsolve(tilde_adj[omega == 0][:, omega == 0],
                          -tilde_adj[omega == 0][:, omega == 1].dot(observed[omega == 1]))
    ).reshape(completed[omega == 0].shape)
    logging.info("Solved!")
    return completed
