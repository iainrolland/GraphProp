import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import svd_flip
from operator import mul
from functools import reduce


def unfold(tensor, mode=0, unfolding="kolda"):
    """
    simply saying tensX of shape (n1, n2, ..., nj) has tensX_(k) of shape
    (nk, -1) does not uniquely define the unfolding, numpy has various reshaping
    options (C-like vs Fortran-like)
    """
    supported = {"kolda": kolda_unfold, "kmode": kmode_unfold, "second_modek": second_modek_unfold}
    if unfolding not in supported:
        raise ValueError(
            "Unfolding of type %s not supported. Choose from %s."
            % (unfolding, list(supported.keys()))
        )
    else:
        return supported[unfolding](tensor, mode)


def fold(matrix, mode, shape, folding="kolda"):
    supported = {"kolda": kolda_fold, "second_modek": second_modek_fold}
    if folding not in supported:
        raise ValueError(
            "Folding of type %s not supported. Choose from %s."
            % (folding, list(supported.keys()))
        )
    else:
        return supported[folding](matrix, mode, shape)


def kolda_unfold(tensor, mode):
    """
    Unfolds a tensors following the Kolda and Bader definition

    Input: tensor of shape (I0, I1, ..., IN)
    Output: matrix of shape (Ik, I0...Ik-1Ik+1...IN)

    Moves the `mode` axis to the beginning and reshapes in Fortran order

    Kolda, T.G. and Bader, B.W., 2009.
    Tensor decompositions and applications.
    SIAM review, 51(3), pp.455-500.
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order="F")


def kolda_fold(matrix, mode, shape):
    """
    Inverse operation of kmode_unfold() (takes matrix and folds back into tensor)

    Kolda, T.G. and Bader, B.W., 2009.
    Tensor decompositions and applications.
    SIAM review, 51(3), pp.455-500.
    """
    return np.moveaxis(np.reshape(matrix,
                                  shape[mode: mode + 1] + shape[:mode] + shape[mode + 1:],
                                  order="F"), 0, mode)


def kmode_unfold(tensor, k):
    """
    Unfolds first k modes of tensor against last N-k modes
    Input: tensor of shape (I0, I1, ..., IN)
    Output: matrix of shape (I0I1...Ik, Ik+1...IN)

    Uses indexing/ordering as given in:
    Yu, J., Zhou, G., Li, C., Zhao, Q. and Xie, S., 2020.
    Low tensor-ring rank completion by parallel matrix factorization.
    IEEE transactions on neural networks and learning systems, 32(7), pp.3020-3033.
    """
    output_shape = (
        reduce(mul, tensor.shape[: k + 1] + (1,)),
        reduce(mul, tensor.shape[k + 1:] + (1,)),
    )
    return np.reshape(tensor, output_shape, order="F")


def second_modek_unfold(tensor, k):
    """
    Unfolds the k-th mode against the others (much like the kolda_unfold()) but the ordering is different
    Input: tensor of shape (I0, I1, ..., IN)
    Output: matrix of shape (Ik, Ik+1...INI0...Ik-1)

    np.transpose(tensor, axes=np.arange(k, k + tensor.ndim) % tensor.ndim) operation cycles the axes such that
    the k-th axis comes first but the ordering is still the same (cyclically)
    """
    return kolda_unfold(np.transpose(tensor, axes=np.arange(k, k + tensor.ndim) % tensor.ndim), 0)


def second_modek_fold(matrix, mode, shape):
    """
    Inverse operation of second_modek_unfold() (takes matrix and folds back into tensor)
    """
    folded = kolda_fold(matrix, 0, tuple(np.roll(shape, -mode)))
    return np.transpose(folded, axes=np.arange(-mode, -mode + len(shape)) % len(shape))


def nuclear_norm(tensor):
    _, sigma, _ = np.linalg.svd(tensor)
    return np.sum(sigma)


def SVT(matrix, tau, rank_prev=None):
    if rank_prev is None:
        sk = 1
    else:
        sk = rank_prev + 1
    U, S, Vh = svd_above_value(matrix, tau, sk)
    # shrink_S = shrinked_sigma(S, tau)
    # rank_prev = np.count_nonzero(shrink_S)
    # diag_shrink_S = np.diag(shrink_S)
    return np.linalg.multi_dot([U, np.diag(S), Vh]), len(S.flatten())


def shrinked_sigma(sigma, tau):
    return np.maximum(sigma - tau, 0)


def truncated_sigma(sigma, tau):
    return np.minimum(sigma, tau)


def spectral_norm(matrix):
    _, S, _ = svds(matrix, k=1)
    return S[0]


def sparse_svd(matrix, k):
    """
    Uses scipy's svd_flip to ensure deterministic output from SVD.

    The columns of U (and rows of Vh) are adjusted such that largest in absolute value
    entries are positive.
    """
    try:
        U, S, Vh = svds(matrix, k=k)
    except ValueError:
        print(k)
        U, S, Vh = svds(matrix, k=k)
    S = S[:: -1]
    U, Vh = svd_flip(U[:, ::-1], Vh[::-1])
    return U, S, Vh


def svd(matrix):
    """
    Uses scipy's svd_flip to ensure deterministic output from SVD.

    The columns of U (and rows of Vh) are adjusted such that largest in absolute value
    entries are positive.
    """
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    U, Vh = svd_flip(U, Vh)
    return U, S, Vh


def svd_above_value(matrix, value, rank_prev=1, return_rank=False):
    """
    Consider only the singular values of size larger than 'value'.

    Returns corresponding cols and rows of U and Vh respectively as well as the values
    themselves (as S).
    """
    U, S, Vh = sparse_svd(matrix, min(rank_prev, min(matrix.shape) - 1))
    while np.min(S) >= value and rank_prev < min(matrix.shape):
        rank_prev = min(rank_prev + 5, min(matrix.shape))
        if rank_prev <= min(matrix.shape) - 1:
            U, S, Vh = sparse_svd(matrix, rank_prev)
        else:
            U, S, Vh = svd(matrix)  # if full rank, cannot use scipy's sparse svd
    rank_prev = np.sum(S > value)
    if not return_rank:
        return U[:, :rank_prev], S[:rank_prev], Vh[:rank_prev]
    else:
        return U[:, :rank_prev], S[:rank_prev], Vh[:rank_prev], rank_prev


def mode_k_product(x, m, mode):
    """
    Performs X \times_{mode} m
    If X \in \mathbb{R}^{I_0\times\ldots I_{N-1}}
    and m \in \mathbb{R}^{J\times I_{mode}}
    then the function returns Y \in \mathbb{R}^{I_0\times\ldots I_{mode-1}\times J\times I_{mode+1}\ldots I_{N-1}}

    Note: gives the same result as fold(m.dot(unfold(x, mode)), mode, (x.shape but with mode-th entry now m.shape[0]))
    """
    x = np.asarray(x)
    m = np.asarray(m)
    if mode < 0 or mode % 1 != 0:
        raise ValueError('`mode` must be a non-negative integer')
    if mode > x.ndim - 1:
        raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
    if m.ndim != 2:
        raise ValueError('Invalid shape of M: {}'.format(m.shape))
    return np.swapaxes(np.swapaxes(x, mode, -1).dot(m.T), mode, -1)
