import argparse
import os
import json

from common import logs
from common import plotting

SUPPORTED_METHODS = ["GraphProp", "GTVM", "AWTC", "HaLRTC"]
SUPPORTED_PATTERNS = ["SLC-off", "partial-overlap"]
EXPERIMENT_FOLDER = "experiments"
INPUTS_FOLDER = "inputs"


def check_arg(arg, supported_args):
    if arg not in supported_args:
        raise ValueError(
            "Unsupported argument. Supported arguments are: {}".format(supported_args)
        )


class Experiment:
    def __init__(self, method, mask, file_0, file_1, iterative=None, neighbours=None):
        self.method = method
        self.mask = mask
        self.neighbours = neighbours
        self.file_0 = file_0
        self.file_1 = file_1
        self.iterative = iterative

    def to_dict(self):
        dictionary = {"mask": self.mask, "method": self.method, "file_0": self.file_0,
                      "file_1": self.file_1}
        if self.iterative is not None:
            dictionary["iterative"] = self.iterative
            dictionary["k"] = self.neighbours
        return dictionary

    def __str__(self):
        return self.to_dict().__str__()


def main(method, pattern, show_results):
    if not os.path.isdir(EXPERIMENT_FOLDER):
        os.mkdir(EXPERIMENT_FOLDER)  # make folder for experiments if it doesn't exist

    file_a = os.path.join(INPUTS_FOLDER, "dayA")
    file_b = os.path.join(INPUTS_FOLDER, "dayB")
    params_path = os.path.join(EXPERIMENT_FOLDER, f"{method.lower()}_demo.json")

    if method in ["GraphProp"]:
        experiment = Experiment(method, pattern, file_a, file_b, neighbours=20, iterative=False)
    else:
        logging.error(f"""Completion method not recognised '{method}'""")
        raise ValueError
    with open(params_path, 'w') as f:
        json.dump(experiment.to_dict(), f)
    output, mask = logs.log_experiment(params_path, overwrite=True)
    if show_results:
        plotting.show_results(params_path, output, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote sensing image completion demo.')
    parser.add_argument('--method', type=str, default='GraphProp')
    parser.add_argument('--pattern', type=str, default='SLC-off')
    parser.add_argument('--plot', type=bool, default=True)
    args = parser.parse_args()

    check_arg(args.method, SUPPORTED_METHODS)
    check_arg(args.pattern, SUPPORTED_PATTERNS)
    main(method=args.method, pattern=args.pattern, show_results=args.plot)
