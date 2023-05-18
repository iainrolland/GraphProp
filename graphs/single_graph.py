import logging

from sklearn.neighbors import kneighbors_graph


def single_graph(image, k=10):
    logging.info(f"Building graph from reference image. Using k={k} nearest neighbours.")

    if not image.ndim == 3:
        msg = f"Expected 3-dimensional image, got {image.ndim} dimensions."
        logging.error(msg)
        raise ValueError(msg)

    a = kneighbors_graph(image.reshape(-1, image.shape[2]), k, include_self=False)
    a = a + a.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
    a[a > 1] = 1  # get rid of any edges we just made double
    return a
