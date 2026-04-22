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


# 1. 기본 데이터에서 필요 Column만 남기기 ===============================

def filter_excel_columns(input_file_path, output_file_path, columns_to_keep):
    """
    엑셀 파일을 읽어 지정된 컬럼만 남기고 나머지는 삭제한 뒤 새로운 파일로 저장합니다.
    """
    try:
        # 1. 엑셀 파일 읽어들임
        df = pd.read_excel(input_file_path)
        
        # (선택 사항) 제공된 리스트의 컬럼이 실제 엑셀 파일에 존재하는지 확인하여 오류 방지
        actual_columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        # 2. 해당 이름을 가진 Column들만 필터링 (나머지 일괄 삭제 효과)
        filtered_df = df[actual_columns_to_keep]
        
        # 3. 결과를 새로운 엑셀 파일로 저장 (인덱스는 제외)
        filtered_df.to_excel(output_file_path, index=False)
        print(f"성공적으로 처리되었습니다. 결과가 '{output_file_path}'에 저장되었습니다.")
        
    except FileNotFoundError:
        print(f"오류: '{input_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

# =====================================================================

# 2. Style Parameter Normalization ====================================

def normalize_MINMAX(data_path, ref_path, target_cols, save_path):
    print(f"[{data_path}] 파일 정규화 작업 시작...")
    
    # 1. 파일 로드
    try:
        df = pd.read_excel(data_path, engine='openpyxl')
        
        # [수정된 부분] index_col=0 옵션을 주어, 불러올 때부터 첫 번째 열을 행 이름(인덱스)으로 고정합니다.
        ref_df = pd.read_excel(ref_path, engine='openpyxl', index_col=0)
    except Exception as e:
        print(f"파일 로드 중 오류가 발생했습니다: {e}")
        return

    # 2. 인덱스(행 이름) 강제 전처리
    # 문자열로 변환 -> 앞뒤 공백 제거 -> 첫 글자만 대문자로 변경 (예: ' MIN ', 'min' -> 모두 'Min'으로 통일)
    ref_df.index = ref_df.index.astype(str).str.strip().str.capitalize()

    # 'Min'과 'Max' 행이 존재하는지 체크 및 디버깅 메시지 추가
    if 'Min' not in ref_df.index or 'Max' not in ref_df.index:
        print("'Min' 또는 'Max' 행을 찾을 수 없습니다.")
        print(f"파이썬이 실제로 읽어들인 행 이름들은 다음과 같습니다: {ref_df.index.tolist()}")
        print("엑셀 파일 첫 열의 데이터가 위 리스트와 다르게 생겼다면 엑셀 파일 확인이 필요합니다.")
        return

    # 3. 지정된 패러미터에 대해 정규화 수행
    normalized_count = 0
    for col in target_cols:
        if col in df.columns and col in ref_df.columns:
            # Min, Max 값 추출
            min_val = float(ref_df.loc['Min', col])
            max_val = float(ref_df.loc['Max', col])
            
            # Max와 Min이 같은 경우 0으로 나누는 에러 방지
            if max_val - min_val == 0:
                print(f" - [경고] {col}: Min과 Max 값이 동일하여 0.0으로 일괄 변환합니다.")
                df[col] = 0.0
            else:
                # 정규화: (X - Min) / (Max - Min)
                df[col] = (df[col] - min_val) / (max_val - min_val)
                
            normalized_count += 1
        else:
            print(f" - [건너뜀] {col}: 데이터 파일이나 기준 파일에 해당 패러미터(Column)가 없습니다.")

    # 4. 정규화된 데이터 저장
    df.to_excel(save_path, index=False, engine='openpyxl')
    
    print(f"\n✅ 정규화 완료! (총 {normalized_count}개 컬럼 적용)")
    print(f" - 저장 위치: {save_path}")

# =====================================================================

# 3. Data Split (Train, Test) =========================================

def split_excel(file_path, train_save_path, test_save_path):
   # 1. 엑셀 파일 로드 (엔진 지정)
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"데이터 로드 완료: 총 {len(df)} 행, {len(df.columns)} 열")
    except Exception as e:
        print(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
        return

    # 3. 데이터 무작위 9:1 분할 (Row 기준)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"데이터 분할 완료: 학습용 {len(train_df)} 행 / 검증용 {len(test_df)} 행")

    # 4. 분할된 데이터를 각각 새로운 엑셀 파일로 저장
    # index=False 로 설정해야 엑셀 파일에 불필요한 인덱스 번호 열이 추가되지 않습니다.
    train_df.to_excel(train_save_path, index=False, engine='openpyxl')
    test_df.to_excel(test_save_path, index=False, engine='openpyxl')
    
    print(f"\n저장 완료!")
    print(f"- 학습용 데이터: {train_save_path}")
    print(f"- 검증용 데이터: {test_save_path}")

# =====================================================================

# 4. Model Design =====================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DataSet Class Setting
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


# Model Architecture Design
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


# Data Load & preprocessing
def prepare_model_data(file_path, test_portion, random_state):
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
        X_raw, y_raw, test_size=test_portion, random_state=random_state
    )
    
    # X(입력)에 대해서만 스케일링 수행
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # y(출력)는 이미 정규화되어 있으므로 그대로 사용!
    y_train_scaled = y_train
    y_val_scaled = y_val
    scaler_y = None # 반환 개수를 맞추기 위해 껍데기만 남겨둠
    
    # 추론 시 복원을 위해 두 스케일러 모두 반환
    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_X, scaler_y


# Model Train
def train_model(FILE_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE,TEST_PORTION, RANDOM_STATE):
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = prepare_model_data(FILE_PATH, TEST_PORTION, RANDOM_STATE)
    
    train_dataset = StyleDifferenceDataset(X_train, y_train)
    val_dataset = StyleDifferenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SiameseStyleRegressor(input_dim=12, output_dim=37).to(device)
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

# =====================================================================

# 5. Evaluation =======================================================

# delta 매개변수: Huber Loss에서 MAE와 MSE를 구분하는 기준값 (기본 1.0)
def evaluate_model(model, scaler_X, test_file_path, ref_file_path, output_file_path="test_results_final.xlsx", delta=1.0):
    print(f"\n--- 전체 데이터 종합 테스트 시작 (MAE, MSE, Huber Loss 평가) ---")

    df_domain = pd.read_excel(ref_file_path, header=None)
    MIN_VALS = df_domain.iloc[2, 1:38].values.astype(float)
    MAX_VALS = df_domain.iloc[3, 1:38].values.astype(float)
    RANGE_VALS = MAX_VALS - MIN_VALS    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  
    
    # 1. 데이터 로드 및 분리
    df_test = pd.read_excel(test_file_path)
    raw_data = df_test.values
    N = len(raw_data)
    
    tgt_indices = np.random.permutation(N)
    tgt_data = raw_data[tgt_indices]
    
    curr_iq_raw = raw_data[:, :12]
    curr_style_norm = raw_data[:, 12:49] # 37차원에 맞춤
    
    tgt_iq_raw = tgt_data[:, :12]
    tgt_style_norm = tgt_data[:, 12:49]
    
    # 2. 입력(X) 정규화 및 추론
    curr_iq_scaled = scaler_X.transform(curr_iq_raw)
    tgt_iq_scaled = scaler_X.transform(tgt_iq_raw)
    
    curr_tensor = torch.tensor(curr_iq_scaled, dtype=torch.float32).to(device)
    tgt_tensor = torch.tensor(tgt_iq_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_diff_norm = model(curr_tensor, tgt_tensor).cpu().numpy()
        
    actual_diff_norm = tgt_style_norm - curr_style_norm
    
    # 3. 자동 로드된 Min/Max 범위를 이용한 물리적 스케일 복원
    pred_diff_raw = pred_diff_norm * RANGE_VALS
    actual_diff_raw = actual_diff_norm * RANGE_VALS
    
    # 4. ★ 3대 오차 지표 계산 (복원된 스케일 기준)
    # 4-1. MAE (절대 오차)
    abs_error = np.abs(pred_diff_raw - actual_diff_raw)
    # 4-2. MSE (제곱 오차)
    sq_error = np.square(pred_diff_raw - actual_diff_raw)
    # 4-3. Huber Loss (임계값 delta 기준)
    huber_error = np.where(abs_error <= delta, 
                           0.5 * sq_error, 
                           delta * abs_error - 0.5 * (delta ** 2))
    
    # 각 지표별 파라미터(37개) 평균 계산
    overall_mae = np.mean(abs_error, axis=0)
    overall_mse = np.mean(sq_error, axis=0)
    overall_huber = np.mean(huber_error, axis=0)
    
    # 5. 엑셀 저장을 위한 데이터 구성
    style_cols = [f"Style_{i+1:02d}" for i in range(37)]
    iq_cols = [f"IQ_{i+1:02d}" for i in range(12)]
    
    rows = []
    
    # [상단 요약] 3가지 오차 지표 전체 평균
    rows.append(["ALL_SUMMARY", "0_MAE_Original"] + [None]*12 + overall_mae.tolist())
    rows.append(["ALL_SUMMARY", "0_MSE_Original"] + [None]*12 + overall_mse.tolist())
    rows.append(["ALL_SUMMARY", f"0_Huber_Original(d={delta})"] + [None]*12 + overall_huber.tolist())
    rows.append([None] * (2 + 12 + 37)) # 빈 줄
    
    # [개별 데이터 분석]
    #for i in range(min(N, 5000)):
    for i in range(N): 
        p_id = f"Pair_{i+1:05d}"
        rows.append([p_id, "1_Actual_Diff_Raw"] + [None]*12 + actual_diff_raw[i].tolist())
        rows.append([p_id, "2_Pred_Diff_Raw"] + [None]*12 + pred_diff_raw[i].tolist())
        rows.append([p_id, "3_Absolute_Error(MAE)"] + [None]*12 + abs_error[i].tolist())
        rows.append([p_id, "4_Squared_Error(MSE)"] + [None]*12 + sq_error[i].tolist())
        rows.append([p_id, "5_Huber_Loss"] + [None]*12 + huber_error[i].tolist())
        rows.append([None] * (2 + 12 + 37)) # 구분선
        
    # 데이터프레임 변환 및 저장
    result_df = pd.DataFrame(rows, columns=["ID", "Type"] + iq_cols + style_cols)
    result_df.to_excel(output_file_path, index=False)
    print(f"✅ 종합 오차(MAE/MSE/Huber) 분석 완료! 결과 저장됨: {output_file_path}")

#===================================================================

# 6. Execute =======================================================
if __name__ == "__main__":

 # 필요 Column만 남기기   
    # 파일 경로
    INPUT_FILE = "IQ_Style_origin.xlsx"       # 원본 엑셀 파일 이름
    OUTPUT_FILE = "IQ_Style_remain.xlsx"     # 저장될 새로운 엑셀 파일 이름
    
    # 남겨야 할 Column
    TARGET_COLUMNS = ["Edge Threshold-Lv2", "Edge Threshold-Lv3", "LapSmoothRate-Lv2"] 
    
    # 필요 Column만 남기기
    filter_excel_columns(INPUT_FILE, OUTPUT_FILE, TARGET_COLUMNS)


# Style Parameter Normalize

    INPUT_NORMAL = "IQ_Style_remain.xlsx"     
    # Min/Max 기준값
    MIN_MAX_REF = "example_minmax_ref.xlsx" 
    # 저장될 파일 이름
    OUTPUT_NORMAL = "IQ_Style_remain_normalized.xlsx"

    # 학습용 데이터 정규화 실행
    normalize_MINMAX(INPUT_NORMAL, MIN_MAX_REF, TARGET_COLUMNS, OUTPUT_NORMAL)

    
# Data Split
    TRAIN_OUTPUT = "IQ_Style_Train.xlsx"
    TEST_OUTPUT = "IQ_Style_Test.xlsx"

    # 함수 실행
    split_excel(OUTPUT_NORMAL, TRAIN_OUTPUT, TEST_OUTPUT)

# HyperParameter Setting
    filepath = TRAIN_OUTPUT  # 엑셀 또는 CSV 파일 경로
    batches = 256
    epochs = 50
    learning_r = 1e-3
    test_portion = 0.2
    random = 42

# 모델 학습 진행 및 객체들 반환 받기
    trained_model, final_scaler_X, final_scaler_y = train_model(filepath, batches, epochs, learning_r, test_portion, random)
    
    # 저장 프로세스
    print("Saving model and scalers...")
    torch.save(trained_model.state_dict(), 'siamese_style_model.pth')
    joblib.dump(final_scaler_X, 'scaler_x.pkl')
    joblib.dump(final_scaler_y, 'scaler_y.pkl')
    print("Save Complete!")

    Test_Result =  "test_results.xlsx"
    evaluate_model(trained_model, final_scaler_X, TEST_OUTPUT, MIN_MAX_REF, Test_Result)
