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
NUM_STYLES = 37 # 전체 Style 파라미터 개수

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
# 4. 데이터 로드 및 전처리
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
    
    # X(입력)에 대해서만 스케일링 수행
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # y(출력)는 이미 정규화되어 있으므로 그대로 사용
    y_train_scaled = y_train
    y_val_scaled = y_val
    scaler_y = None 
    
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_X, scaler_y

# ==========================================
# 5. 개별 모델 학습 함수 (1개의 파라미터만 학습하도록 수정)
# ==========================================
def train_single_model(X_train, X_val, y_train_single, y_val_single, param_idx):
    train_dataset = StyleDifferenceDataset(X_train, y_train_single)
    val_dataset = StyleDifferenceDataset(X_val, y_val_single)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 생성 시 output_dim을 1로 지정 (원형 클래스의 구조 유지)
    model = SiameseStyleRegressor(input_dim=12, output_dim=1).to(device)
    criterion = nn.HuberLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
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
        
        # 화면 출력이 너무 길어지는 것을 방지하기 위해 10 Epoch 마다 출력
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  └ Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model

# ==========================================
# 6. 실행 및 모델/스케일러 일괄 저장
# ==========================================
if __name__ == '__main__':
    # 1. 전체 데이터 준비
    X_train, X_val, y_train, y_val, final_scaler_X, final_scaler_y = prepare_data(FILE_PATH)
    
    # 2. 공통 스케일러 저장 (모델 수와 상관없이 1번만 저장하면 됨)
    print("Saving common scalers...")
    joblib.dump(final_scaler_X, 'scaler_x.pkl')
    joblib.dump(final_scaler_y, 'scaler_y.pkl')
    
    # 3. 모델을 저장할 디렉토리 생성
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 37개의 파라미터에 대해 순차적으로 모델 학습 및 저장
    print(f"Starting Training for {NUM_STYLES} separate models...")
    for i in range(NUM_STYLES):
        print(f"\n[{i+1}/{NUM_STYLES}] Training model for Style Parameter {i+1}...")
        
        # i번째 파라미터의 타겟값만 추출 (N x 1 차원 유지)
        y_train_single = y_train[:, i:i+1]
        y_val_single = y_val[:, i:i+1]
        
        # 개별 모델 학습 수행
        trained_model = train_single_model(X_train, X_val, y_train_single, y_val_single, param_idx=i)
        
        # 개별 모델 저장
        model_filename = os.path.join(save_dir, f'siamese_style_model_param_{i+1}.pth')
        torch.save(trained_model.state_dict(), model_filename)
        print(f"▶ Saved Model {i+1} to {model_filename}")
        
    print("\nAll 37 models have been successfully trained and saved!")