import csv
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

from torch.utils.data import random_split, Dataset, DataLoader

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

    # Fix length
    len_data = x.shape[1]
    if len_data > N_SAMPLES:
        random_index = random.randint(0, len_data - N_SAMPLES)
        x = x[:, random_index : random_index + N_SAMPLES]
    else:
        len_missing = N_SAMPLES - len_data
        x = F.pad(x, (0, len_missing))

    # # Melspectrogram
    # x = mel_spectrogram(x)
    x = x.unsqueeze(1)

    return x


class UrbanSound8KDataset(Dataset):
    CLASS_ID_TO_NAME = [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music",
    ]

    NAME_TO_CLASS_ID = {name: i for i, name in enumerate(CLASS_ID_TO_NAME)}

    def __init__(self, root: str = os.path.join("data", "UrbanSound8K"), pretraining: bool = False) -> None:

        # Double check the user has passed a valid dataset path
        if os.path.isdir(root):
            self.root = root
        else:
            raise OSError(f"{root} is not a valid folder.")

        self.pretraining = pretraining

        # Open the metadata file
        with open(os.path.join(self.root, "metadata", "UrbanSound8K.csv"), "r", newline="") as f:
            reader = csv.reader(f)
            metadata = list(reader)[1:]

        # Get the full file_paths, and labels
        self.file_paths = [os.path.join(self.root, "audio", f"fold{m[5]}", m[0]) for m in metadata]
        self.class_ids = torch.LongTensor([self.NAME_TO_CLASS_ID[m[7]] for m in metadata])

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


class ESC50Dataset(Dataset):
    def __init__(self, root: str = os.path.join("data", "ESC-50-master"), pretraining: bool = False):
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

        self.file_paths = [os.path.join(self.audio_dir, fname) for fname in df["filename"].tolist()]
        self.class_ids = df["target"].tolist()
        self.n_folds = df["fold"].unique().tolist()
        self.CLASS_ID_TO_NAME = dict(set(zip(df["target"], df["category"])))

    def __getitem__(self, index):
        x, sample_rate = torchaudio.load(self.file_paths[index])
        x = preprocess(x, sample_rate)
        y = self.class_ids[index]
        return x, y

    def __len__(self):
        return len(self.file_paths)

    @property
    def n_classes(self) -> int:
        return len(self.CLASS_ID_TO_NAME)

    @property
    def class_names(self):
        return self.CLASS_ID_TO_NAME


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        train_ratio: float,
        shuffle: bool = True,
        num_workers: int = 0,
        pretraining: bool = False,
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

        dataset = (
            UrbanSound8KDataset(pretraining=pretraining)
            if dataset_name == "urbansound8k"
            else ESC50Dataset(pretraining=pretraining)
        )
        print(f"There are {len(dataset)} samples in the {dataset_name} dataset.")
        signal, _ = dataset[1]

        self.height = signal.size(-2)
        self.width = signal.size(-1)

        self.n_classes = dataset.n_classes
        self.class_names = dataset.class_names

        # TODO: k-folds validation
        train_len = int(len(dataset) * self.train_ratio)
        val_len = len(dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(dataset, [train_len, val_len])

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
