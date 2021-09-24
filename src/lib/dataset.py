import csv
import os
from pathlib import Path

import torchaudio.transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class UrbanSound8KDataset(Dataset):
    def __init__(self,
                 root: str = '../../UrbanSound8K',
                 transformation: torchaudio.transforms.MelSpectrogram = None,
                 target_sample_rate: int = 22050,
                 n_samples: int = 22050,
                 device: str = 'cpu',
                 should_ignore_cache=True,
                 should_make_cache=False):

        if os.path.isdir(root):
            # assume user has passed a valid dataset to us
            self.root = root
        else:
            raise OSError(f'{root} is not a valid folder.')

        # open metadata CSV file with Pandas
        # self.metadata = pd.read_csv(f'{self.root}/metadata/UrbanSound8K.csv')
        with open(f'{self.root}/metadata/UrbanSound8K.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            self.metadata = list(reader)[1:]

        if transformation is None:
            self.transformation = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )
        else:
            self.transformation = transformation

        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
        self.device = device
        self.should_ignore_cache = should_ignore_cache
        self.should_make_cache = should_make_cache

        if self.should_make_cache:
            # make directory for transforms to save them for later use
            self.tensor_cache_dir = f'{self.root}/tensor_cache'
            if not os.path.isdir(self.tensor_cache_dir):
                os.mkdir(self.tensor_cache_dir)
        else:
            self.tensor_cache_dir = ''

    def __getitem__(self, index) -> T_co:
        item_meta = self.metadata[index]

        filename = item_meta[0]
        label = item_meta[7]
        foldnum = item_meta[5]

        # Check for tensor cache
        tensor_cache_path = f'{self.tensor_cache_dir}/{filename}.pt'
        if not self.should_ignore_cache and os.path.isfile(tensor_cache_path):
            t_data = torch.load(tensor_cache_path)
        else:
            wav, sr = torchaudio.load(f'{self.root}/audio/fold{foldnum}/{filename}')

            t_data = wav.to(self.device)
            t_data = self._resample(t_data, self.target_sample_rate)
            t_data = self._mix_down(t_data)
            t_data = self._fix_length(t_data)
            t_data = self.transformation(t_data)

            if self.should_make_cache:
                # Save the tensor for later
                torch.save(t_data, Path(tensor_cache_path))

        return t_data, label

    def __len__(self):
        return len(self.metadata)

    def _resample(self, t_data, sample_rate):
        if sample_rate != self.target_sample_rate:
            resample = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            return resample(t_data)
        else:
            return t_data

    def _fix_length(self, t_data):
        len_data = t_data.shape[1]
        if len_data > self.n_samples:
            return t_data[:, :self.n_samples]
        elif len_data < self.n_samples:
            len_missing = self.n_samples - len_data
            return torch.nn.functional.pad(t_data, (0, len_missing))
        else:
            return t_data

    @staticmethod
    def _mix_down(t_data):
        if t_data.shape[0] > 1:
            return torch.mean(t_data, dim=0, keepdim=True)
        else:
            return t_data

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == "__main__":
    """
    Random benchmarking stuff here.
    """
    import time
    import random

    dataset = UrbanSound8KDataset(should_make_cache=True, should_ignore_cache=False)
    indices = random.sample(range(0, len(dataset)), len(dataset))

    t0 = time.time()
    for i in indices:
        data, lbl = dataset[i]
    t1 = time.time()

    print(f'Total time: {t1 - t0}')
