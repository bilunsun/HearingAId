from train import Model
from lib import dataset

import pytorch_lightning as pl
import torch

import argparse
import random
import time
import statistics
import shutil


def main(args: argparse.ArgumentParser):
    configs = vars(args)

    # Set the seed; pl.seed_everything sets a random seed if args.seed is None
    seed = pl.seed_everything(configs["seed"])
    configs["seed"] = seed

    ds = dataset.UrbanSound8KDataset()
    width = ds[0][0].size(-2)
    height = ds[0][0].size(-1)

    configs["width"] = width
    configs["height"] = height
    configs["n_classes"] = ds.n_classes

    model = Model(**configs)

    infer_times = []

    print("\nStarting benchmark...")

    for it, i in enumerate(random.sample(range(len(ds)), args.times)):
        sample = torch.reshape(ds[i][0], (1, 1, width, height))

        # time inference section
        start_time = time.time()
        model(sample)
        elapsed_time = time.time() - start_time

        infer_times.append(elapsed_time)

        printProgressBar(it + 1, args.times)

    print(
        f"Ran {args.times} samples."
        + f"\n\tAverage: {statistics.mean(infer_times)}"
        + f"\n\tStandard Deviation: {statistics.stdev(infer_times)}"
        + f"\n\tMax: {max(infer_times)}"
    )


def printProgressBar(iteration, total, prefix="", suffix="", fill="█", barfill=" ", printEnd="\r"):
    w, h = shutil.get_terminal_size()
    length = w - 9 - len(str(total)) * 2

    percent = f"{int(100 * (iteration / total)) : >3}"
    filledLength = int(length * iteration // total)

    bar = fill * filledLength + barfill * (length - filledLength)

    print(f"\r{percent}%|{bar}| {iteration}/{total}", end=printEnd)

    if iteration == total:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--times", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
