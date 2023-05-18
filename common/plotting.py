import matplotlib.pyplot as plt
import json
import numpy as np

from utils import get_file, normalise


def show_results(params_path, output, mask):
    with open(params_path, 'r') as f:
        params = json.load(f)

    ground_truth = np.stack([get_file(params["file_0"])[0], get_file(params["file_1"])[0]], axis=-1)

    # make copy of ground truth to remove values from
    image = ground_truth.copy()

    # remove values where mask is False
    image[~mask] = np.nan  # remove ground truth from masked areas

    # normalise images
    image, low, high = normalise(image)
    ground_truth, _, _ = normalise(ground_truth, low, high)
    # insert 0.0 where mask is False (we used nan before for computing percentiles in normalisation step)
    image = np.nan_to_num(image, nan=0.)

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].set_ylabel("Day A")
    ax[1, 0].set_ylabel("Day B")

    ax[0, 0].imshow(ground_truth[..., [2, 1, 0], 0])
    ax[0, 0].set_title("Ground truth")
    ax[1, 0].imshow(ground_truth[..., [2, 1, 0], 1])

    ax[0, 1].imshow(image[..., [2, 1, 0], 0])
    ax[0, 1].set_title("Inputs")
    ax[1, 1].imshow(image[..., [2, 1, 0], 1])

    ax[0, 2].set_title(f"{params['method']}\nOutputs")
    ax[0, 2].imshow(output[..., [2, 1, 0], 0])
    ax[1, 2].imshow(output[..., [2, 1, 0], 1])
    plt.show()
