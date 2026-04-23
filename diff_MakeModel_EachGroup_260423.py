import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

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
        
        # 2. 확장된 델타 회귀기 (Delta Regressor)
        self.regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(128, output_dim) # 최종 출력
        )

    def forward(self, current_iq, target_iq):
        curr_feat = self.encoder(current_iq)
        tgt_feat = self.encoder(target_iq)
        
        feat_diff = tgt_feat - curr_feat
        style_diff_pred = self.regressor(feat_diff)
        return style_diff_pred

# ==========================================
# 4. 데이터 로드 및 전처리
# ==========================================
def prepare_data(file_path):
    print("Loading data...")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Column 명칭 추출 (출력 파라미터 분류용)
    y_cols = df.columns[12:].tolist()
    
    X_raw = df.iloc[:, 0:12].values
    y_raw = df.iloc[:, 12:].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # y는 정규화되어 있다고 가정
    y_train_scaled = y_train
    y_val_scaled = y_val
    scaler_y = None 
    
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_X, scaler_y, y_cols

# ==========================================
# 5. 그룹별 개별 모델 학습 함수
# ==========================================
def train_group_model(X_train, X_val, y_train_group, y_val_group, output_dim, group_name):
    print(f"\n--- Training Model for Group: [{group_name}] (Output Dim: {output_dim}) ---")
    
    train_dataset = StyleDifferenceDataset(X_train, y_train_group)
    val_dataset = StyleDifferenceDataset(X_val, y_val_group)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 해당 그룹의 파라미터 개수에 맞춰 output_dim 설정
    model = SiameseStyleRegressor(input_dim=12, output_dim=output_dim).to(device)
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
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  └ Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model

# ==========================================
# 6. 실행 및 모델/스케일러 일괄 저장
# ==========================================
if __name__ == '__main__':
    # 1. 전체 데이터 준비 및 Column 이름 가져오기
    X_train, X_val, y_train, y_val, final_scaler_X, final_scaler_y, y_cols = prepare_data(FILE_PATH)
    
    # 2. 출력 파라미터 분류 로직 적용 (5개 그룹)
    groups = {
        'Lv0': [],
        'Lv1': [],
        'Lv2': [],
        'Lv3': [],
        'Others': []
    }
    
    for idx, col_name in enumerate(y_cols):
        col_str = str(col_name)
        if col_str.endswith(' Lv0'):
            groups['Lv0'].append(idx)
        elif col_str.endswith(' Lv1'):
            groups['Lv1'].append(idx)
        elif col_str.endswith(' Lv2'):
            groups['Lv2'].append(idx)
        elif col_str.endswith(' Lv3'):
            groups['Lv3'].append(idx)
        else:
            groups['Others'].append(idx)
            
    # 그룹 분류 결과 출력
    print("\n[Parameter Grouping Results]")
    for g_name, indices in groups.items():
        print(f" - {g_name}: {len(indices)} parameters")
    
    # 3. 저장소 생성 및 스케일러 저장 (공통 1회)
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nSaving common scaler...")
    joblib.dump(final_scaler_X, os.path.join(save_dir, 'scaler_x.pkl'))
    if final_scaler_y is not None:
        joblib.dump(final_scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))
    
    # 4. 분류된 5개 그룹에 대해 순차적으로 모델 학습 및 저장
    print("\nStarting Training for Grouped Models...")
    for group_name, param_indices in groups.items():
        num_params = len(param_indices)
        
        if num_params == 0:
            print(f"\nSkipping [{group_name}] (No parameters found for this group).")
            continue
            
        # 해당 그룹의 타겟값만 슬라이싱하여 추출
        y_train_group = y_train[:, param_indices]
        y_val_group = y_val[:, param_indices]
        
        # 개별 그룹 모델 학습 수행 (output_dim을 파라미터 개수만큼 동적 지정)
        trained_model = train_group_model(
            X_train, X_val, 
            y_train_group, y_val_group, 
            output_dim=num_params, 
            group_name=group_name
        )
        
        # 모델 저장
        model_filename = os.path.join(save_dir, f'siamese_style_model_group_{group_name}.pth')
        torch.save(trained_model.state_dict(), model_filename)
        print(f"▶ Saved Model for [{group_name}] to {model_filename}")
        
    print("\nAll group models have been successfully trained and saved!")