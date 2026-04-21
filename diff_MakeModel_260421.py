import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # 스케일러 저장을 위해

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
FILE_PATH = 'data.xlsx'  # 엑셀 또는 CSV 파일 경로
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 2. 데이터셋 클래스 정의
# ==========================================
class StyleDifferenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.num_samples = len(X)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx1 = idx
        idx2 = np.random.randint(0, self.num_samples)
        
        current_iq = self.X[idx1]
        target_iq = self.X[idx2]
        
        style_diff = self.y[idx2] - self.y[idx1]
        
        return current_iq, target_iq, style_diff

# ==========================================
# 3. 모델 아키텍처 정의 (수정 금지 부분)
# ==========================================
class SiameseStyleRegressor(nn.Module):
    def __init__(self, input_dim=12, output_dim=37, dropout_rate=0.2):
        super(SiameseStyleRegressor, self).__init__()
        
        # 1. 확장된 공유 인코더 (Shared Encoder)
        # 기존: 12 -> 64 -> 64
        # 변경: 12 -> 128 -> 256 -> 256 (학습 용량 대폭 증대)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 과적합 방지
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        # 2. 확장된 델타 회귀기 (Delta Regressor)
        # 상호작용 계층에서 넘어온 256차원(Difference)을 입력받음
        # 기존: 64 -> 128 -> 71
        # 변경: 256 -> 256 -> 128 -> 37 (정규화된 패러미터 개수에 맞춤)
        self.regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(128, output_dim) # 최종 출력: 37차원 (0~1 정규화값 차이)
        )

    def forward(self, current_iq, target_iq):
        # A. 개별 인코딩 (동일한 가중치 공유)
        curr_feat = self.encoder(current_iq)
        tgt_feat = self.encoder(target_iq)
        
        # B. 상호작용 계층: 단순 뺄셈 유지 (Target - Current)
        feat_diff = tgt_feat - curr_feat
        
        # C. 최종 예측: 37종 Style 파라미터 변화량 산출
        style_diff_pred = self.regressor(feat_diff)
        return style_diff_pred

# ==========================================
# 4. 데이터 로드 및 전처리 (scaler_X, scaler_y 모두 반환하도록 수정)
# ==========================================
def prepare_data(file_path):
    print("Loading data...")
    # 파일 확장자에 따라 다르게 로드
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    X_raw = df.iloc[:, 0:12].values
    y_raw = df.iloc[:, 12:].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # 추론 시 복원을 위해 두 스케일러 모두 반환
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_X, scaler_y

# ==========================================
# 5. 학습 루프 (반환값 수정)
# ==========================================
def train_model():
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = prepare_data(FILE_PATH)
    
    train_dataset = StyleDifferenceDataset(X_train, y_train)
    val_dataset = StyleDifferenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SiameseStyleRegressor(input_dim=12, hidden_dim=64, output_dim=71).to(device)
    criterion = nn.HuberLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for curr_iq, tgt_iq, actual_diff in train_loader:
            curr_iq, tgt_iq, actual_diff = curr_iq.to(device), tgt_iq.to(device), actual_diff.to(device)
            optimizer.zero_grad()
            pred_diff = model(curr_iq, tgt_iq)
            loss = criterion(pred_diff, actual_diff)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for curr_iq, tgt_iq, actual_diff in val_loader:
                curr_iq, tgt_iq, actual_diff = curr_iq.to(device), tgt_iq.to(device), actual_diff.to(device)
                pred_diff = model(curr_iq, tgt_iq)
                loss = criterion(pred_diff, actual_diff)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("Training Complete!")
    return model, scaler_X, scaler_y

# ==========================================
# 6. 실행 및 모델/스케일러 저장
# ==========================================
if __name__ == '__main__':
    # 모델 학습 진행 및 객체들 반환 받기
    trained_model, final_scaler_X, final_scaler_y = train_model()
    
    # 저장 프로세스
    print("Saving model and scalers...")
    torch.save(trained_model.state_dict(), 'siamese_style_model.pth')
    joblib.dump(final_scaler_X, 'scaler_x.pkl')
    joblib.dump(final_scaler_y, 'scaler_y.pkl')
    print("Save Complete!")