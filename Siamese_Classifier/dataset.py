import torch
import numpy as np
from torch.utils.data import Dataset

class StyleDifferenceDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        self.num_samples = len(X)
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        tgt_idx = np.random.randint(0, self.num_samples)
        return self.X[idx], self.X[tgt_idx], self.y[tgt_idx] - self.y[idx]\n