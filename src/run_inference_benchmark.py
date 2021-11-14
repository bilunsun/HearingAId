from train import Model
from lib import dataset, AudioDataModule

import pytorch_lightning as pl
import torch

import argparse
import random
import time
import statistics


WIDTH = 64
HEIGHT = 44


def main(args: argparse.ArgumentParser):
    configs = vars(args)

    # Set the seed; pl.seed_everything sets a random seed if args.seed is None
    seed = pl.seed_everything(configs["seed"])
    configs["seed"] = seed

    audio_datamodule = AudioDataModule(**configs)
    configs["width"] = audio_datamodule.width
    configs["height"] = audio_datamodule.height
    configs["n_classes"] = audio_datamodule.n_classes

    model = Model(**configs)
    ds = dataset.UrbanSound8KDataset()

    infer_times = []

    for i in random.sample(range(len(ds)), args.times):
        sample = torch.reshape(ds[i][0], (1, 1, WIDTH, HEIGHT))
        # import pdb
        # pdb.set_trace()

        # time inference section
        start_time = time.time()
        model(sample)
        elapsed_time = time.time() - start_time

        infer_times.append(elapsed_time)

    print(f'Ran {args.times} samples.\n\tAverage: {statistics.mean(infer_times)}' +
          f'\n\tStandard Deviation: {statistics.stdev(infer_times)}' +
          f'\n\tMax: {max(infer_times)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--times', type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)

    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
