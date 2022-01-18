import csv
import os
import pytorch_lightning as pl
import torch
import torchaudio.transforms

from pathlib import Path
from torch.utils.data import random_split, Dataset
from tqdm.auto import tqdm


import math
import random


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.it = list(range(math.ceil(len(self.dataset) / self.batch_size)))

    def __len__(self):
        return len(self.it)

    def __iter__(self):
        it = self.it[:]

        if self.shuffle:
            random.shuffle(it)

        for i in it:
            yield self.dataset[i * self.batch_size : (i + 1) * self.batch_size]


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

    def __init__(
        self,
        root: str = os.path.join("data", "UrbanSound8K"),
        transformation: torchaudio.transforms.MelSpectrogram = None,
        target_sample_rate: int = 22050,
        n_samples: int = 22050,
    ):

        if os.path.isdir(root):
            # assume user has passed a valid dataset to us
            self.root = root
        else:
            raise OSError(f"{root} is not a valid folder.")

        self.target_sample_rate = target_sample_rate
        self.n_samples = n_samples

        self.transformation = transformation or torchaudio.transforms.MelSpectrogram(sample_rate=target_sample_rate, n_fft=1024, hop_length=256, n_mels=128)

        # Open the metadata file
        with open(os.path.join(self.root, "metadata", "UrbanSound8K.csv"), "r", newline="") as f:
            reader = csv.reader(f)
            metadata = list(reader)[1:]

        # Get the full file_paths, and labels
        file_paths = [os.path.join(self.root, "audio", f"fold{m[5]}", m[0]) for m in metadata]
        labels = [m[7] for m in metadata]

        # Make directory for transforms to save them for later use
        self.tensor_cache_dir = os.path.join(self.root, "tensor_cache")
        if not os.path.isdir(self.tensor_cache_dir):
            os.mkdir(self.tensor_cache_dir)
            self.preprocess(file_paths, labels)
        elif not os.listdir(self.tensor_cache_dir):
            self.preprocess(file_paths, labels)

        # Load the dataset into memory
        print("Loading dataset into memory...")
        signals = []
        class_ids = []
        for filename in tqdm(os.listdir(self.tensor_cache_dir)):
            class_id = int(filename.split("__")[0])
            signal = torch.load(os.path.join(self.tensor_cache_dir, filename))

            signals.append(signal)
            class_ids.append(class_id)
        print("Done loading dataset into memory...")

        self.signals = torch.cat(signals).unsqueeze(1)  # Need an extra dim for channels
        self.class_ids = torch.LongTensor(class_ids)

    def preprocess(self, file_paths, labels):
        print("Preprocessing...")

        for filename, label in tqdm(zip(file_paths, labels), total=len(labels)):
            wav, _ = torchaudio.load(filename)

            t_data = self._resample(wav, self.target_sample_rate)
            t_data = self._mix_down(t_data)
            t_data_list = self._fix_length(t_data)
            t_data_list = [self.transformation(t_data) for t_data in t_data_list]

            # Save the tensor in the cache directory
            basename = os.path.basename(filename)
            class_id = self.NAME_TO_CLASS_ID[label]

            for i, t_data in enumerate(t_data_list):
                cache_path = os.path.join(self.tensor_cache_dir, f"{class_id}__{basename}_{i}.pt")
                torch.save(t_data, Path(cache_path))

        print("Done preprocessing.")

    def __getitem__(self, index):
        return self.signals[index], self.class_ids[index]

    def __len__(self):
        return len(self.class_ids)

    def _resample(self, t_data, sample_rate):
        if sample_rate != self.target_sample_rate:
            resample = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            return resample(t_data)
        else:
            return t_data

    def _fix_length(self, t_data):
        len_data = t_data.shape[1]
        if len_data > self.n_samples:
            t_data_list = [t_data[:, i:i+self.n_samples] for i in range(len_data // self.n_samples)]
            return t_data_list
        elif len_data < self.n_samples:
            len_missing = self.n_samples - len_data
            return [torch.nn.functional.pad(t_data, (0, len_missing))]
        else:
            return [t_data]

    @staticmethod
    def _mix_down(t_data):
        if t_data.shape[0] > 1:
            return torch.mean(t_data, dim=0, keepdim=True)
        else:
            return t_data

    @property
    def n_classes(self) -> int:
        return len(self.CLASS_ID_TO_NAME)

    @property
    def class_names(self):
        return self.CLASS_ID_TO_NAME


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, shuffle: bool, num_workers: int, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        dataset = UrbanSound8KDataset()
        print(f"There are {len(dataset)} samples in the dataset.")
        signal, _ = dataset[1]

        self.width = signal.size(-2)
        self.height = signal.size(-1)

        self.n_classes = dataset.n_classes
        self.class_names = dataset.class_names
        print("N classes", self.n_classes)

        train_len = int(len(dataset) * 0.8)
        val_len = len(dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(dataset, [train_len, val_len])
        print("OK")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
