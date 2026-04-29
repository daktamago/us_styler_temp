import torch
from torch.utils.data import Dataset
import numpy as np

class StyleDifferenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.num_samples = len(X)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx1 = idx
        # 랜덤하게 비교 대상을 하나 뽑음
        idx2 = np.random.randint(0, self.num_samples)
        
        current_iq = self.X[idx1]
        target_iq = self.X[idx2]
        
        # Style 변화량(Difference) 계산
        style_diff = self.y[idx2] - self.y[idx1]
        
        return current_iq, target_iq, style_diff