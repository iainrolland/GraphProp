import numpy as np
from tqdm import tqdm
import logging

import common.tensor_utils as tu


def complete(incomplete_tensor, mask, rho=.0001, K=200, epsilon=1e-2, T=.85, eta=50, t=1.15):
    logging.info(f"Starting AWTC with rho={rho}, K={K}, epsilon={epsilon}, T={T}, eta={eta}, t={t}")
    # initialisation
    X = incomplete_tensor.copy()
    X[mask == 0] = 0.
    dim = incomplete_tensor.shape
    M = np.zeros(np.insert(dim, 0, len(dim)))
    Y = np.zeros(np.insert(dim, 0, len(dim)))
    order = incomplete_tensor.ndim

    alphas = np.array([1] * order) / order  # initialise weights

    patience = 0
    last_err = None

    # iterate
    try:
        for k in tqdm(range(K)):
            X_prev = X.copy()
            for i in range(order):
                M[i] = tu.fold(tu.SVT(tu.unfold(X + Y[i] / rho, i), alphas[i] / rho)[0], i, X.shape)
            X[mask == 0] = (np.sum(M - Y / rho, axis=0) / order)[mask == 0]
            Y = Y - rho * (M - np.broadcast_to(X, np.insert(dim, 0, len(dim))))
            alphas = weights(X, T, eta)  # update weights

            err = np.linalg.norm(X - X_prev) / np.linalg.norm(X_prev)
            if last_err is not None and err < last_err:
                patience += 1
            else:
                patience = 0
            last_err = err * 1.

            if err < epsilon and patience > 10:
                logging.info(f"Converged after just {k} of maximum {K} iterations")
                break

            rho *= t
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Returning current result.")
        print("Interrupted by user. Returning current result.")
    return X


def k_norm(unfolding: np.ndarray, T):
    r"""
    \hat{k}_i in notation of original paper Eq. (10) and (11)
    """
    if not unfolding.ndim == 2:
        raise ValueError(f"Function takes unfolded tensors (i.e. of dim 2) but was given array of dim {unfolding.ndim}")
    sigma = np.linalg.svd(unfolding.T,
                          full_matrices=False,
                          compute_uv=False)  # sorted in DESCENDING order (largest first)
    # sigma = sigma[::-1]  # ASCENDING order (smallest first)
    return (1 + np.min(np.argwhere((np.cumsum(sigma) / np.sum(sigma)) >= T).flatten())) / len(sigma)


def weights(tensor, T, eta):
    """
    implements Eq. (12) from original paper
    """
    k = np.array([k_norm(tu.unfold(tensor, i), T) for i in range(tensor.ndim)])
    w = np.exp(eta * k / np.sum(k))
    w = w / sum(w)
    return w
