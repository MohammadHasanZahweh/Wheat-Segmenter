from torch.utils.data import Dataset

import os
import numpy as np
from torch.utils.data import Dataset

class PixelRangeNPYDataset(Dataset):
    """
    Loads .npy files for classification where:
    - dataset folder contains one subfolder per class
    - each class folder contains files named N.npy
    - load only files whose index is in [k1, k2]
    """
    def __init__(self, root, k1, k2, transform=None):
        self.root = root
        self.k1 = k1
        self.k2 = k2
        self.transform = transform
        self.samples = []  # (path, class_idx)

        class_names = sorted(os.listdir(root))
        for class_idx, cname in enumerate(class_names):
            cpath = os.path.join(root, cname)
            if not os.path.isdir(cpath):
                continue

            for fname in os.listdir(cpath):
                if fname.endswith(".npy"):
                    idx = int(os.path.splitext(fname)[0])
                    if k1 <= idx <= k2:
                        self.samples.append((os.path.join(cpath, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)   # shape [n, c, t] or something similar

        if self.transform:
            x = self.transform(x)

        return x, label


class PixelFromKNPYDataset(Dataset):
    """
    Loads .npy files for classification where:
    - dataset folder contains one subfolder per class
    - each class folder has files named N.npy
    - load only files whose index >= k
    """
    def __init__(self, root, k, transform=None):
        self.root = root
        self.k = k
        self.transform = transform
        self.samples = []

        class_names = sorted(os.listdir(root))
        for class_idx, cname in enumerate(class_names):
            cpath = os.path.join(root, cname)
            if not os.path.isdir(cpath):
                continue

            for fname in os.listdir(cpath):
                if fname.endswith(".npy"):
                    idx = int(os.path.splitext(fname)[0])
                    if idx >= k:
                        self.samples.append((os.path.join(cpath, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)

        if self.transform:
            x = self.transform(x)

        return x, label


if __name__ == "__main__":
    from pathlib import Path
    PROCESS_DATA_PATH       = Path(r"processed_data/")
    PIXEL_SPLIT_DATA_PATH   = PROCESS_DATA_PATH/"split_processed_data/wheat"

    train_data = PixelRangeNPYDataset(PIXEL_SPLIT_DATA_PATH, 0, 0)
    print(len(train_data))

    val_data = PixelRangeNPYDataset(PIXEL_SPLIT_DATA_PATH, 1,1)
    print(len(val_data))

    test_data = PixelFromKNPYDataset(PIXEL_SPLIT_DATA_PATH, 2)
    print(len(test_data))


