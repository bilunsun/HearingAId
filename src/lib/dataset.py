import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader
from typing import List

import random


TARGET_SAMPLE_RATE = 22_050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
N_SAMPLES = 22_050

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)


def preprocess(x, sample_rate):
    # Reshape
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    elif x.size(1) == 1 or x.size(1) == 2:
        x = x.T

    # Resample
    if sample_rate != TARGET_SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        x = resample(x)

    # Mixdown
    if x.size(0) > 1:
        x = torch.mean(x, dim=0, keepdim=True)

    # Fix length by padding zeros on left and right sides, then crop
    len_data = x.shape[1]
    if len_data < N_SAMPLES:
        len_missing = int((N_SAMPLES - len_data) * 1.2)
        left_pad = len_missing // 2
        right_pad = len_missing - left_pad
        x = F.pad(x, (left_pad, right_pad))

    len_data = x.shape[1]
    random_index = random.randint(0, len_data - N_SAMPLES)
    x = x[:, random_index : random_index + N_SAMPLES]

    x = x.unsqueeze(1)

    return x


class StandardDataset(Dataset):
    """
    Simple base class for Urbansound8K and ESC-50 datasets
    """

    def __getitem__(self, index):
        return self._regular_getitem(index) if not self.pretraining else self._pretraining_getitem(index)

    def _regular_getitem(self, index):
        x, sample_rate = torchaudio.load(self.file_paths[index])
        x = preprocess(x, sample_rate)
        y = self.class_ids[index]
        return x, y

    def _pretraining_mix(self, x1, x2, y1, y2):
        r = random.uniform(0, 1)

        numerator = r * x1 + (1 - r) * x2
        denominator = (r ** 2 + (1 - r) ** 2)**0.5

        x_mixed = numerator / denominator
        y_mixed = r * y1 + (1 - r) * y2

        return x_mixed, y_mixed

    def _pretraining_getitem(self, _):
        index1, index2 = random.choices(range(len(self)), k=2)

        x1, y1 = self._regular_getitem(index1)
        y1 = F.one_hot(y1, num_classes=self.n_classes)

        x2, y2 = self._regular_getitem(index2)
        y2 = F.one_hot(y2, num_classes=self.n_classes)

        x_mixed, y_mixed = self._pretraining_mix(x1, x2, y1, y2)

        return x_mixed, y_mixed

    def __len__(self):
        return len(self.file_paths)

    @property
    def n_classes(self) -> int:
        return len(self.CLASS_ID_TO_NAME)

    @property
    def class_names(self):
        return self.CLASS_ID_TO_NAME

    @classmethod
    def split_folds(cls, dataset: Dataset, n_validation_folds: int = 1):
        val_folds = random.choices(dataset.n_folds, k=n_validation_folds)
        train_folds = list(set(dataset.n_folds) - set(val_folds))
        train_set = cls(pretraining=dataset.pretraining, folds=train_folds)
        val_set = cls(pretraining=dataset.pretraining, folds=val_folds)

        return train_set, val_set, train_folds, val_folds


class UrbanSound8KDataset(StandardDataset):
    def __init__(self, root: str = os.path.join("data", "UrbanSound8K"), pretraining: bool = False, folds: List[int] = None) -> None:

        # Double check the user has passed a valid dataset path
        if os.path.isdir(root):
            self.root = root
        else:
            raise OSError(f"{root} is not a valid folder.")

        self.pretraining = pretraining

        self.audio_dir = os.path.join(self.root, "audio")
        self.meta_path = os.path.join(self.root, "metadata", "UrbanSound8K.csv")

        df = pd.read_csv(self.meta_path)

        if folds is not None:
            df = df[df["fold"].isin(folds)]

        self.file_paths = [os.path.join(self.audio_dir, f"fold{f}", fname) for f, fname in zip(df["fold"], df["slice_file_name"])]
        self.class_ids = torch.LongTensor(df["classID"].tolist())
        self.n_folds = df["fold"].unique().tolist()
        self.CLASS_ID_TO_NAME = dict(set(zip(df["classID"], df["class"])))


class ESC50Dataset(StandardDataset):
    def __init__(self, root: str = os.path.join("data", "ESC-50-master"), pretraining: bool = False, folds: List[int] = None) -> None:
        super().__init__()

        # Double check the user has passed a valid dataset path
        if os.path.isdir(root):
            self.root = root
        else:
            raise OSError(f"{root} is not a valid folder.")

        self.pretraining = pretraining

        self.audio_dir = os.path.join(self.root, "audio")
        self.meta_path = os.path.join(self.root, "meta", "esc50.csv")

        df = pd.read_csv(self.meta_path)

        if folds is not None:
            df = df[df["fold"].isin(folds)]

        self.file_paths = [os.path.join(self.audio_dir, fname) for fname in df["filename"].tolist()]
        self.class_ids = torch.LongTensor(df["target"].tolist())
        self.n_folds = df["fold"].unique().tolist()
        self.CLASS_ID_TO_NAME = dict(set(zip(df["target"], df["category"])))


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        train_ratio: float = 0.9,
        pretraining: bool = False,
        folds: List[int] = None,
        **kwargs,
    ):
        super().__init__()

        assert 0 < train_ratio < 1

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pretraining = pretraining
        self.extra_kwargs = {"prefetch_factor": 4, "persistent_workers": True} if self.num_workers > 0 else {}

        dataset_class = UrbanSound8KDataset if dataset_name == "urbansound8k" else ESC50Dataset
        dataset = dataset_class(pretraining=pretraining, folds=folds)
        print(f"There are {len(dataset)} samples in the {dataset_name} dataset.")
        signal, _ = dataset[1]

        self.height = signal.size(-2)
        self.width = signal.size(-1)

        self.n_classes = dataset.n_classes
        self.class_names = dataset.class_names

        # k-folds validation
        self.train_dataset, self.val_dataset, self.train_folds, self.val_folds = dataset_class.split_folds(dataset)
        print(f"Train folds:", self.train_folds)
        print(f"Val folds:", self.val_folds)
        print(f"There are {len(self.train_dataset)} training samples.")
        print(f"There are {len(self.val_dataset)} validation samples.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            **self.extra_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            **self.extra_kwargs,
        )


def test():
    dm = AudioDataModule(dataset_name="urbansound8k", batch_size=32, train_ratio=0.9, shuffle=False, num_workers=0, pretraining=True)
    for i, (x, y) in enumerate(dm.train_dataloader()):
        print(x.shape, x.mean(), x.std())
        print(y.shape)

        import pdb; pdb.set_trace()

        if i == 5:
            exit()


if __name__ == "__main__":
    test()
