import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader, random_split
from typing import List

import random


TARGET_SAMPLE_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
N_SAMPLES = 24_000

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)


def preprocess(
    x,
    sample_rate,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    n_samples: int = N_SAMPLES,
    convert_to_mel: bool = False,
):
    # Normalize
    x = x / 32768

    # Resample
    if sample_rate != target_sample_rate:
        resample = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        x = resample(x)

    # Mixdown
    if x.size(1) > 1:
        x = torch.mean(x, dim=1, keepdim=True)

    # Fix length by padding zeros on left and right sides, then crop
    len_data = x.shape[-1]
    if len_data < n_samples:
        len_missing = int((n_samples - len_data) * 1.2)
        left_pad = len_missing // 2
        right_pad = len_missing - left_pad
        x = F.pad(x, (left_pad, right_pad))

    len_data = x.shape[-1]
    random_index = random.randint(0, len_data - n_samples)
    x = x[:, :, random_index : random_index + n_samples]

    if convert_to_mel:
        x = mel_spectrogram(x.squeeze(0))

    return x


class StandardDataset(Dataset):
    """
    Simple base class for Urbansound8K and ESC-50 datasets
    """

    def __init__(
        self, root: str, target_sample_rate: int, n_samples: int, pretraining: str, convert_to_mel: bool
    ) -> None:
        self.GETITEM_METHOD = {
            None: self._regular_getitem,
            "mix": self._mix_getitem,
            "simsiam": self._simsiam_getitem,
        }

        # Double check the user has passed a valid dataset path
        if os.path.isdir(root):
            self.root = root
        else:
            raise OSError(f"{root} is not a valid folder.")

        self.root = root
        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples
        self.pretraining = pretraining
        self.convert_to_mel = convert_to_mel

        self.getitem_method = self.GETITEM_METHOD[pretraining]

    def __getitem__(self, index):
        return self.getitem_method(index)

    def _scale(self, x):
        """Slow down or speed up"""
        scale_factor = random.uniform(0.8, 1.25)
        x = F.interpolate(x, scale_factor=scale_factor, recompute_scale_factor=True)  # Without recompute, get warning
        return x

    def _gain(self, x):
        """Gain +/- 6dB, or 3.981"""
        gain_factor = random.uniform(-3.981, 3.981)
        x = x * gain_factor
        return x

    def _regular_getitem(self, index: int, scale_augment: bool = True, gain_augment: bool = True):
        x, sample_rate = torchaudio.load(self.file_paths[index], normalize=False)

        # Reshape
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        elif x.size(1) == 1 or x.size(1) == 2:
            x = x.T

        # Add extra dim and convert to float
        x = x.unsqueeze(0).float()

        # Pre-augment
        if scale_augment:
            x = self._scale(x)

        x = preprocess(x, sample_rate, self.target_sample_rate, self.n_samples, self.convert_to_mel)

        # Post-augment
        if gain_augment:
            x = self._gain(x)

        y = self.class_ids[index]
        return x, y

    def _pretraining_mix(self, x1, x2, y1, y2):
        r = random.uniform(0, 1)

        numerator = r * x1 + (1 - r) * x2
        denominator = (r ** 2 + (1 - r) ** 2) ** 0.5

        x_mixed = numerator / denominator
        y_mixed = r * y1 + (1 - r) * y2

        return x_mixed, y_mixed

    def _mix_getitem(self, _):
        index1, index2 = random.choices(range(len(self)), k=2)

        x1, y1 = self._regular_getitem(index1, gain_augment=False)
        y1 = F.one_hot(y1, num_classes=self.n_classes)

        x2, y2 = self._regular_getitem(index2, gain_augment=False)
        y2 = F.one_hot(y2, num_classes=self.n_classes)

        x_mixed, y_mixed = self._pretraining_mix(x1, x2, y1, y2)
        x_mixed = self._gain(x_mixed)

        return x_mixed, y_mixed

    def _simsiam_getitem(self, index):
        """Generates two random augments of the same sample"""
        x1, _ = self._regular_getitem(index)
        x2, _ = self._regular_getitem(index)

        return x1, x2

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
        # val_folds = random.choices(dataset.n_folds, k=n_validation_folds)
        val_folds = [3]
        train_folds = list(set(dataset.n_folds) - set(val_folds))
        train_set = cls(
            pretraining=dataset.pretraining,
            target_sample_rate=dataset.target_sample_rate,
            n_samples=dataset.n_samples,
            convert_to_mel=dataset.convert_to_mel,
            folds=train_folds,
        )
        val_set = cls(
            pretraining=dataset.pretraining,
            target_sample_rate=dataset.target_sample_rate,
            n_samples=dataset.n_samples,
            convert_to_mel=dataset.convert_to_mel,
            folds=val_folds,
        )

        return train_set, val_set, train_folds, val_folds


class UrbanSound8KDataset(StandardDataset):
    def __init__(
        self,
        root: str = os.path.join("data", "UrbanSound8K"),
        pretraining: str = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        n_samples: int = N_SAMPLES,
        convert_to_mel: bool = False,
        folds: List[int] = None,
    ) -> None:
        super().__init__(root, target_sample_rate, n_samples, pretraining, convert_to_mel)

        self.audio_dir = os.path.join(self.root, "audio")
        self.meta_path = os.path.join(self.root, "metadata", "UrbanSound8K.csv")

        df = pd.read_csv(self.meta_path)

        if folds is not None:
            df = df[df["fold"].isin(folds)]

        self.file_paths = [
            os.path.join(self.audio_dir, f"fold{f}", fname) for f, fname in zip(df["fold"], df["slice_file_name"])
        ]
        self.class_ids = torch.LongTensor(df["classID"].tolist())
        self.n_folds = df["fold"].unique().tolist()
        self.CLASS_ID_TO_NAME = dict(set(zip(df["classID"], df["class"])))


class ESC50Dataset(StandardDataset):
    def __init__(
        self,
        root: str = os.path.join("data", "ESC-50-master"),
        pretraining: str = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        n_samples: int = N_SAMPLES,
        convert_to_mel: bool = False,
        folds: List[int] = None,
    ) -> None:
        super().__init__(root, target_sample_rate, n_samples, pretraining, convert_to_mel)

        self.audio_dir = os.path.join(self.root, "audio")
        self.meta_path = os.path.join(self.root, "meta", "esc50.csv")

        df = pd.read_csv(self.meta_path)

        if folds is not None:
            df = df[df["fold"].isin(folds)]

        self.file_paths = [os.path.join(self.audio_dir, fname) for fname in df["filename"].tolist()]
        self.class_ids = torch.LongTensor(df["target"].tolist())
        self.n_folds = df["fold"].unique().tolist()
        self.CLASS_ID_TO_NAME = dict(set(zip(df["target"], df["category"])))


class AudioSetDataset(StandardDataset):
    N_SECONDS = 4

    def __init__(
        self,
        root: str = os.path.join("data", "AudioSet"),
        pretraining: str = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        n_samples: int = N_SAMPLES,
        folds: List[int] = None,
        convert_to_mel: bool = False,
    ) -> None:

        """
        WARNING: OVERWRITES N_SAMPLES TO BE 5X THE TARGET_SAMPLE_RATE TO USE 5 SECONDS OF AUDIO
        Also, only use the following classes:
        classes = [
            "acoustic_guitar",
            "alarm_clock",
            "bell",
            "bird",
            "brass_instrument",
            "car_alarm",
            "cat",
            "dog",
            "doorbell",
            "drum_kit",
            "explosion",
            "helicopter",
            "honking",
            "laughter",
            "plucked_string_instrument",
            "police_siren",
            "rapping",
            "reversing_beeps",
            "silence",
            "singing",
            "speech",
            "telephone_ring",
            "train_horn",
            "water",
        ]
        """
        super().__init__(root, target_sample_rate, int(target_sample_rate * self.N_SECONDS), pretraining, convert_to_mel)


        self.root = root
        # classes = [name for name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, name))]
        classes = [
            "acoustic_guitar",
            "alarm_clock",
            "bell",
            "bird",
            "brass_instrument",
            "car_alarm",
            "cat",
            "dog",
            "doorbell",
            "drum_kit",
            "explosion",
            "helicopter",
            "honking",
            "laughter",
            "plucked_string_instrument",
            "police_siren",
            "rapping",
            "reversing_beeps",
            "silence",
            "singing",
            "speech",
            "telephone_ring",
            "train_horn",
            "water",
        ]
        self.CLASS_ID_TO_NAME = dict(zip(range(len(classes)), classes))

        self.file_paths = []
        self.class_ids = []

        for i, c in enumerate(classes):
            class_files = [os.path.join(self.root, c, f) for f in os.listdir(os.path.join(self.root, c))]
            self.file_paths.extend(class_files)
            self.class_ids.extend([i] * len(class_files))

        self.class_ids = torch.LongTensor(self.class_ids)
        self.n_folds = None

    # Overwrite split_folds for AudioSet
    @classmethod
    def split_folds(cls, dataset: Dataset, n_validation_folds: int = 1):
        train_len = int(len(dataset) * 0.9)
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])

        return train_set, val_set, None, None


class CustomDataset(StandardDataset):

    def __init__(
        self,
        root: str = os.path.join("data", "collected_data"),
        pretraining: str = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        n_samples: int = N_SAMPLES,
        folds: List[int] = None,
        convert_to_mel: bool = False,
    ) -> None:
        super().__init__(root, target_sample_rate, n_samples, pretraining, convert_to_mel)

        self.root = root

        self.file_paths = [os.path.join(self.root, f) for f in os.listdir(self.root)]
        classes = sorted(list(set([f.split("_")[0] for f in os.listdir(self.root)])))

        self.CLASS_ID_TO_NAME = dict(zip(range(len(classes)), classes))
        print(self.CLASS_ID_TO_NAME)

        self.class_ids = [classes.index(f.split("_")[0]) for f in os.listdir(self.root)]

        self.class_ids = torch.LongTensor(self.class_ids)
        self.n_folds = None

    # Overwrite split_folds for AudioSet
    @classmethod
    def split_folds(cls, dataset: Dataset, n_validation_folds: int = 1):
        train_len = int(len(dataset) * 0.9)
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])

        return train_set, val_set, None, None


class AudioDataModule(pl.LightningDataModule):
    NAME_TO_DATASET_CLASS = {
        "urbansound8k": UrbanSound8KDataset,
        "esc50": ESC50Dataset,
        "audioset": AudioSetDataset,
        "custom": CustomDataset
    }

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pretraining: str = None,
        convert_to_mel: bool = False,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        n_samples: int = N_SAMPLES,
        folds: List[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pretraining = pretraining
        self.convert_to_mel = convert_to_mel
        self.extra_kwargs = {"prefetch_factor": 4, "persistent_workers": True} if self.num_workers > 0 else {}

        dataset_class = self.NAME_TO_DATASET_CLASS[dataset_name]
        dataset = dataset_class(pretraining=pretraining, folds=folds, target_sample_rate=target_sample_rate, n_samples=n_samples)
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
    dm = AudioDataModule(
        dataset_name="custom",
        batch_size=1,
        shuffle=False,
        num_workers=0,
        target_sample_rate=16_000,
        n_samples=16_000 * 4,
    )
    for i, (x, y) in enumerate(dm.train_dataloader()):
        print(x.shape, x.mean(), x.std())
        print(y.shape)

        import pdb

        pdb.set_trace()

        if i == 5:
            exit()


if __name__ == "__main__":
    test()
