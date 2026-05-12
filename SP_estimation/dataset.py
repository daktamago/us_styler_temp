import torch
from torch.utils.data import Dataset

class StyleDirectDataset(Dataset):
    def __init__(self, X, y):
        # X: IQ parameters, y: Style parameters
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.num_samples = len(X)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 쌍을 만들지 않고 해당 인덱스의 데이터를 그대로 반환
        iq_input = self.X[idx]
        style_target = self.y[idx]
        
        return iq_input, style_target