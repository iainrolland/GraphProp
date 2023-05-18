import numpy as np


def never_observed_check(adj, x, with_reversing_mask=True):
    if adj.shape[0] != x.shape[0]:
        raise ValueError("Shapes incompatible")
    degree = adj.sum(axis=1)
    if np.any(degree == 0):
        ever_seen = np.where(degree != 0)[0]
        if with_reversing_mask:
            return adj[ever_seen][:, ever_seen], x[ever_seen, ...], np.array(degree != 0).squeeze()
        else:
            return adj[ever_seen][:, ever_seen], x[ever_seen, ...]
    else:
        if with_reversing_mask:
            return adj, x, np.arange(x.shape[0])
        else:
            return adj, x
