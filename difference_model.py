import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
FILE_PATH = 'data.csv'  # 엑셀 파일인 경우 'data.xlsx'로 변경하고 pd.read_excel 사용
BATCH_SIZE = 256        # RTX 3050 VRAM을 고려한 배치 사이즈 (128~512 권장)
EPOCHS = 50
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2         # 검증 데이터 20% 이하
RANDOM_STATE = 42

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 2. 데이터셋 클래스 정의 (동적 페어링)
# ==========================================
class StyleDifferenceDataset(Dataset):
    def __init__(self, X, y):
        """
        X: 정규화된 IQ 파라미터 (N, 12)
        y: 정규화된 Style 파라미터 (N, 71)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.num_samples = len(X)

    def __len__(self):
        # 1 에포크당 원본 데이터 크기만큼 학습을 진행 (필요시 배수로 늘릴 수 있음)
        return self.num_samples

    def __getitem__(self, idx):
        # idx1은 순차적으로, idx2는 랜덤하게 선택하여 동적으로 쌍(Pair)을 생성
        idx1 = idx
        idx2 = np.random.randint(0, self.num_samples)
        
        current_iq = self.X[idx1]
        target_iq = self.X[idx2]
        
        # 목표: Target Style - Current Style (Difference)
        style_diff = self.y[idx2] - self.y[idx1]
        
        return current_iq, target_iq, style_diff

# ==========================================
# 3. 모델 아키텍처 정의 (Siamese Network)
# ==========================================
class SiameseStyleRegressor(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=71):
        super(SiameseStyleRegressor, self).__init__()
        
        # Shared Encoder (공유 가중치 적용)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Delta Regressor (차이값을 바탕으로 최종 Difference 예측)
        # Interaction Layer 방식으로 단순 Difference(차이)를 활용하여 대칭성 확보
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, current_iq, target_iq):
        # 1. 각각 동일한 인코더 통과
        curr_feat = self.encoder(current_iq)
        tgt_feat = self.encoder(target_iq)
        
        # 2. 특징 벡터의 차이 계산 (Interaction)
        feat_diff = tgt_feat - curr_feat
        
        # 3. 최종 Style Parameter Difference 예측
        style_diff_pred = self.regressor(feat_diff)
        return style_diff_pred

# ==========================================
# 4. 데이터 로드 및 전처리
# ==========================================
def prepare_data(file_path):
    print("Loading data...")
    # csv 대신 excel을 사용할 경우: df = pd.read_excel(file_path)
    # 더미 데이터 생성 (테스트용, 실제 적용시 삭제)
    # df = pd.DataFrame(np.random.randn(1000, 83)) 
    df = pd.read_csv(file_path)
    
    # 1~12열(index 0~11)은 Input(IQ), 13열~(index 12~)은 Output(Style)
    X_raw = df.iloc[:, 0:12].values
    y_raw = df.iloc[:, 12:].values
    
    # Train / Val 분할 (데이터가 섞이는 것을 방지하기 위해 분할 먼저 수행)
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 정규화 (스케일링) - 필수
    scaler_X = StandardScaler()
    scaler_y = StandardScaler() # y 차이값의 범위를 맞추기 위해 Y도 스케일링
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_y

# ==========================================
# 5. 학습 및 검증 루프
# ==========================================
def train_model():
    # 데이터 준비
    X_train, X_val, y_train, y_val, scaler_y = prepare_data(FILE_PATH)
    
    train_dataset = StyleDifferenceDataset(X_train, y_train)
    val_dataset = StyleDifferenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 초기화
    model = SiameseStyleRegressor(input_dim=12, hidden_dim=64, output_dim=71).to(device)
    
    # 손실 함수 및 옵티마이저 (Huber Loss 권장 및 AdamW)
    criterion = nn.HuberLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 학습 스케줄러 (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for curr_iq, tgt_iq, actual_diff in train_loader:
            curr_iq, tgt_iq, actual_diff = curr_iq.to(device), tgt_iq.to(device), actual_diff.to(device)
            
            optimizer.zero_grad()
            
            # 예측 (Mixed Precision 사용시 torch.cuda.amp.autocast() 추가 가능)
            pred_diff = model(curr_iq, tgt_iq)
            
            # 손실 계산 및 역전파
            loss = criterion(pred_diff, actual_diff)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # 검증 루프
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
    return model, scaler_y

if __name__ == '__main__':
    # 실제 데이터셋 파일(CSV 또는 Excel)이 준비되면 아래 함수를 실행하세요.
    # model, scaler = train_model()
    pass