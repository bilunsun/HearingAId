import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):

    def __init__(self, **kwargs):
        dataset_size = 10_000
        self.x = torch.randn(dataset_size, 1, 64, 64)
        self.y = torch.randint(low=0, high=6, size=(dataset_size, ))

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

        self.train_dataset = AudioDataset(**kwargs)
        self.val_dataset = AudioDataset(**kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
