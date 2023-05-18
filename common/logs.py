import logging
import os
import json
import datetime
import numpy as np

from utils import get_file, normalise, masking, completing
from common import evaluate


def add_logging_level(level_name, level_num, method_name=None):
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        raise AttributeError('{} already defined in logging module'.format(level_name))
    if hasattr(logging, method_name):
        raise AttributeError('{} already defined in logging module'.format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError('{} already defined in logger class'.format(method_name))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)


def log_experiment(path, overwrite=False):
    log_path = path.replace(".json", "_0.log")
    if os.path.isfile(log_path):
        if not overwrite:
            i = 1
            while os.path.isfile(log_path):
                log_path = path.replace(".json", f"_{i}.log")
                i += 1
        else:
            os.remove(log_path)

    # Clear all handlers
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)

    # Set up logging
    logging.basicConfig(filename=log_path, filemode='a',
                        format="%(asctime)s.%(msecs)03d >> %(levelname)s >> %(message)s",
                        level=logging.INFO)
    logging.Formatter.formatTime = (
        lambda self, record, datefmt=None: datetime.datetime.fromtimestamp(
            record.created, datetime.timezone.utc
        ).astimezone().isoformat(sep="T", timespec="milliseconds")
    )

    # Run experiment
    with open(path, 'r') as f:
        output, mask = instigate(json.load(f))
    # Save results
    np.save(path.replace(".json", "_output.npy"), output.astype(np.float32))
    np.save(path.replace(".json", "_mask.npy"), mask.astype(bool))
    return output, mask


def instigate(params):
    logging.info("Run-time parameters:")
    logging.info(params)
    logging.info("Experiment starting...")

    ground_truth = np.stack([get_file(params["file_0"])[0], get_file(params["file_1"])[0]], axis=-1)

    # get mask (bool, True if observed False otherwise)
    mask = masking(ground_truth[..., 0].shape, params)

    # make copy of ground truth to remove values from
    image = ground_truth.copy()

    # remove values where mask is False
    image[~mask] = np.nan  # remove ground truth from masked areas

    # normalise images
    image, low, high = normalise(image)
    ground_truth, _, _ = normalise(ground_truth, low, high)
    # insert 0.0 where mask is False (we used nan before for computing percentiles in normalisation step)
    image = np.nan_to_num(image, nan=0.)

    # get prediction
    output = completing(image, mask, params)

    # assess prediction
    assess_completion(output, ground_truth, mask)

    return output, mask


def assess_completion(prediction, gt, mask):
    never_seen = np.sum(mask, axis=-1, keepdims=True) == 0
    loss_mask = ~mask.copy()  # compute loss where mask is False
    loss_mask *= ~never_seen
    logging.info(f"Per-band, file_0 loss computed over {np.sum(loss_mask[..., 0], axis=(0, 1))} pixels")
    logging.info(f"Per-band, file_1 loss computed over {np.sum(loss_mask[..., 1], axis=(0, 1))} pixels")
    logging.info(f"MSE: {evaluate.mse(gt, prediction, loss_mask)}")
    logging.info(f"RMSE: {evaluate.rmse(gt, prediction, loss_mask)}")
    logging.info(f"MAE {evaluate.mae(gt, prediction, loss_mask)}")
    mpsnr = evaluate.mean_psnr(
        np.concatenate([gt[..., 0][loss_mask[..., 0]], gt[..., 1][loss_mask[..., 1]]], axis=0),
        np.concatenate([prediction[..., 0][loss_mask[..., 0]], prediction[..., 1][loss_mask[..., 1]]], axis=0)
    )
    logging.info(f"mPSNR: {mpsnr}")
