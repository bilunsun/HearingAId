# import sys
# sys.path.insert("../../data")

import os
from torch.utils.data import Dataset
from torchaudio.datasets import YESNO


class YesNoDataset(Dataset):
    DIR_NAME = "waves_yesno"
    def __init__(self, root: str = "data"):
        self.dataset = YESNO(root, download=True)


if __name__ == "__main__":
    dataset = YesNoDataset()
