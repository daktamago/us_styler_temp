import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 데이터셋 클래스 정의 (1 row -> 1 row)
# ==========================================
class DirectStyleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.num_samples = len(X)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 기존처럼 2개를 뽑아 차이를 구하지 않고, 해당 인덱스의 입력과 정답만 반환
        inputs = self.X[idx]
        targets = self.y[idx]
        return inputs, targets

# ==========================================
# 2. 모델 아키텍처 정의 (Siamese 제거, 단일 입력 멀티 헤드)
# ==========================================
class DirectStyleRegressor(nn.Module):
    def __init__(self, input_dim=12, head_dims=[12, 12, 13], dropout_rate=0.2):
        super(DirectStyleRegressor, self).__init__()
        
        # 1. 인코더 (단일 입력을 처리)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        # 2. 공통 특성 추출기
        self.shared_regressor_base = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        # 3. 멀티 헤드 (지정된 개수만큼 분할 출력)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, out_dim)
            ) for out_dim in head_dims
        ])

    def forward(self, x):
        # A. 단일 데이터 인코딩
        feat = self.encoder(x)
        
        # B. 공통 회귀 베이스 통과 (Siamese의 차이 연산 부분 삭제됨)
        shared_feat = self.shared_regressor_base(feat)
        
        # C. 멀티 헤드 통과 및 결합
        head_outputs = [head(shared_feat) for head in self.heads]
        style_pred = torch.cat(head_outputs, dim=1) 
        
        return style_pred

# ==========================================
# 3. 학습 루프 (입력 변수 단순화)
# ==========================================
def train_direct_model(file_path, batch_size=256, epochs=50, lr=1e-3):
    # prepare_data 함수는 이전과 동일하게 사용 (X, y 스케일링 로직 동일)
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = prepare_data(file_path)
    
    train_dataset = DirectStyleDataset(X_train, y_train)
    val_dataset = DirectStyleDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 선언 (단일 입력 모델로 변경)
    model = DirectStyleRegressor(input_dim=12, head_dims=[15, 12, 10]).to(device)
    criterion = nn.HuberLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("Starting Direct Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 언패킹 변수 2개로 축소 (inputs, targets)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("Training Complete!")
    return model, scaler_X, scaler_y