import csv
import os
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

    # Melspectrogram
    x = mel_spectrogram(x)

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

    def __init__(self, root: str = os.path.join("data", "UrbanSound8K")) -> None:

        # Double check the user has passed a valid dataset path
        if os.path.isdir(root):
            self.root = root
        else:
            raise OSError(f"{root} is not a valid folder.")

        # Open the metadata file
        with open(os.path.join(self.root, "metadata", "UrbanSound8K.csv"), "r", newline="") as f:
            reader = csv.reader(f)
            metadata = list(reader)[1:]

        # Get the full file_paths, and labels
        self.file_paths = [os.path.join(self.root, "audio", f"fold{m[5]}", m[0]) for m in metadata]
        self.class_ids = torch.LongTensor([self.NAME_TO_CLASS_ID[m[7]] for m in metadata])

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
    def __init__(self, batch_size: int, train_ratio: float, shuffle: bool = True, num_workers: int = 0, **kwargs):
        super().__init__()

        assert 0 < train_ratio < 1

        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.extra_kwargs = {"prefetch_factor": 4, "persistent_workers": True} if self.num_workers > 0 else {}

        dataset = UrbanSound8KDataset()
        print(f"There are {len(dataset)} samples in the dataset.")
        signal, _ = dataset[1]

        self.height = signal.size(-2)
        self.width = signal.size(-1)

        self.n_classes = dataset.n_classes
        self.class_names = dataset.class_names

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
    dm = AudioDataModule(batch_size=32, train_ratio=0.9, shuffle=False, num_workers=0)
    for x, y in dm.train_dataloader():
        print(x.shape)
        print(y.shape)
        exit()


if __name__ == "__main__":
    test()
