import os
import xarray as xr
import numpy as np
import logging

import masks
import graphs
from NgEtAl2017 import AWTC
from LiuEtAl2012 import HaLRTC
import gtvm
import diffusion


def get_file(path):
    """
    Opens array from .netcdf file.

    Returns a tuple of (bands, qa) where bands is 7-band image and qa is the quality assurance band
    """
    if os.path.isfile(path):
        data = xr.open_dataset(path)
        bands, qa = np.split(
            np.stack([data.data_vars[key].to_numpy() for key in data.data_vars.keys() if key != "spatial_ref"], axis=-1)
            , [-1], axis=-1)
        bands = bands.astype(float)  # move band axis to last
        return bands, qa
    else:  # if path invalid
        raise ValueError


def normalise(image, low=None, high=None):
    """
    Per band normalisation of images.
    Outputs an array with values between 0 and 1.

    If low and high are not provided, they are computed as the 1st and 99th percentile of the pixel values.
    If they are provided then they are used as the normalisation bounds.
    Normalise that band in all images to lie between 0 and 1 using computed/provided low/high.

    Note: we assume nan values where data has been masked out, these are ignored in the percentile computation.
    """
    if image.ndim != 4:
        msg = f"image has {image.ndim} dimensions, expected 4"
        logging.error(msg)
        raise ValueError(msg)
    if low is None or high is None:
        low, high = np.nanpercentile(image, q=[1, 99], axis=(0, 1, 3), keepdims=True)
    logging.info(
        f"Images normalised with 0 for each band as {np.squeeze(low)} and 1.0 for each band as {np.squeeze(high)}"
    )
    return np.clip((image - low) / (high - low), 0, 1), low, high


def masking(acquision_array_shape, params):
    """
    returns array of shape acquision_array_shape + (2,) where extra dim comes from stacking temporally
    """
    if params["mask"] == "SLC-off":
        return masks.slc_off_mask(acquision_array_shape)
    elif params["mask"] == "partial-overlap":
        logging.info("First 90 rows and columns of file_0 masked")
        logging.info("Last 90 rows and columns of file_1 masked")
        return masks.partial_overlap(acquision_array_shape, 90, 90, 90, 90)
    else:
        logging.error(f"""Masking method not recognised '{params["mask"]}'""")
        raise ValueError


def build_graph(image, mask, params):
    graph = graphs.dual_graph(image, mask, k=params["k"])
    logging.info("Graph computed!")
    return graph


def completing(image, mask, params):
    """
    returns array of shape same as image (ndim = 4, last dim is temporal)
    """
    if params["method"] == "AWTC":
        return AWTC.complete(image, mask, epsilon=1e-2, rho=1e-4, eta=10)
    elif params["method"] == "HaLRTC":
        return HaLRTC.complete(image, mask, epsilon=1e-2, rho=1e-4)
    elif params["method"] in ["GTVM", "GraphProp"]:
        adj = build_graph(image, mask, params)
        if params["method"] == "GTVM":
            return np.stack(
                [gtvm.complete(adj, image[..., 0], mask[..., 0], iterative=params["iterative"]),
                 gtvm.complete(adj, image[..., 1], mask[..., 1], iterative=params["iterative"])],
                axis=-1
            )
        else:
            return np.stack(
                [diffusion.graph_prop(adj, image[..., 0], mask[..., 0], iterative=params["iterative"]),
                 diffusion.graph_prop(adj, image[..., 1], mask[..., 1], iterative=params["iterative"])],
                axis=-1
            )
    else:
        logging.error(f"""Completion method not recognised '{params["method"]}'""")
        raise ValueError
