import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):

    def __init__(self, **kwargs):
        self.x = torch.randn(1_000, 1, 64, 64)
        self.y = torch.randint(low=0, high=6, size=(1_000, 1))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class AudioDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, shuffle: bool, num_workers: int, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_set = AudioDataset(**kwargs)
        self.val_set = AudioDataset(**kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
