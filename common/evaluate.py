import numpy as np


def mae(truth, prediction, mask=None):
    """
    Computes the mean absolute error between truth and prediction
    Average performed across all bands if multi-bands
    When `mask' is not None then mae returned only for pixels where `mask == True'
    """
    ae = np.abs(truth - prediction)
    if mask is not None:
        ae = ae[mask]
    return np.mean(ae)


def psnr(truth, prediction, mask=None):
    """
    Computes the peak signal-to-noise ration (dB)
    Average performed across all bands if multi-bands
    When `mask' is not None then mae returned only for pixels where `mask == True'

    returns float if truth.ndim == 2 else np.array if truth.ndim == 3 (where each value is the PSNR for that band)
    """
    check_shapes(truth, prediction)
    if mask is not None:
        mask = check_mask(mask)
        truth_flat = truth[mask]  # truth flat (#samples,) if truth.ndim == 2 or (#samples, #feats) if truth.ndim == 3
        pred_flat = prediction[mask]
    else:
        truth_flat = truth.copy()
        pred_flat = prediction.copy()
    peak = np.max(truth_flat, axis=0)
    pred_rmse = np.mean((pred_flat - truth_flat) ** 2, axis=0) ** 0.5
    return 20 * np.log10(peak / pred_rmse)


def mean_psnr(truth, prediction, mask=None):
    return np.mean(psnr(truth, prediction, mask))


def mse(truth, prediction, mask=None):
    """
    Computes the mean square error between truth and prediction
    Average performed across all bands if multi-bands
    When `mask' is not None then mae returned only for pixels where `mask == True'
    """
    se = (truth - prediction) ** 2
    if mask is not None:
        se = se[mask]
    return np.mean(se)


def check_shapes(truth, prediction):
    if truth.shape != prediction.shape:
        raise ValueError(
            f"Shape of `truth' {truth.shape} must match shape of `prediction' {prediction.shape}"
        )


def check_mask(mask):
    if not mask.dtype == bool:
        mask = mask.astype(bool)
    return mask


def rmse(truth, prediction, mask=None):
    """
    Computes the RMS error between truth and prediction
    Average performed across all bands if multi-bands
    When `mask' is not None then mae returned only for pixels where `mask == True'
    """
    return mse(truth, prediction, mask) ** 0.5
