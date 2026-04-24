import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. 모델 아키텍처 정의 (학습 시와 동일해야 함)
# ==========================================
class SiameseStyleRegressor(nn.Module):
    def __init__(self, input_dim=12, output_dim=37, dropout_rate=0.2):
        super(SiameseStyleRegressor, self).__init__()
        
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
        
        self.regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, output_dim)
        )

    def forward(self, current_iq, target_iq):
        curr_feat = self.encoder(current_iq)
        tgt_feat = self.encoder(target_iq)
        feat_diff = tgt_feat - curr_feat
        return self.regressor(feat_diff)

# ==========================================
# 2. 통합 테스트 및 결과 저장 함수
# ==========================================
def evaluate_grouped_models(model_dir, scaler_path, test_file_path, ref_file_path, output_file_path="test_results_grouped_5.xlsx", delta=1.0):
    print(f"\n--- 5개 그룹 모델 순차 테스트 및 결과 통합 시작 ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 공통 데이터 및 스케일러 로드
    scaler_X = joblib.load(scaler_path)
    df_test = pd.read_excel(test_file_path)
    df_domain = pd.read_excel(ref_file_path, header=None)
    
    # 물리적 스케일 복원을 위한 범위 값 추출
    MIN_VALS = df_domain.iloc[2, 1:38].values.astype(float)
    MAX_VALS = df_domain.iloc[3, 1:38].values.astype(float)
    RANGE_VALS = MAX_VALS - MIN_VALS
    
    raw_data = df_test.values
    N = len(raw_data)
    
    # 2. 파라미터 그룹 분류 (학습 때와 동일한 기준 적용)
    # y값에 해당하는 12번 인덱스 이후의 컬럼명들 추출
    y_cols = df_test.columns[12:49].tolist() 
    
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
            
    print("\n[Parameter Grouping for Testing]")
    for g_name, indices in groups.items():
        print(f" - {g_name}: {len(indices)} parameters")
        
    # 3. 입력 및 타겟 데이터 준비
    curr_iq_raw = raw_data[:, :12]
    curr_style_norm = raw_data[:, 12:49]
    
    # 정확한 오차 비교를 위해 페어링 인덱스 고정
    perm_idx = np.random.permutation(N)
    tgt_iq_raw = raw_data[perm_idx, :12]
    tgt_style_norm = raw_data[perm_idx, 12:49]
    
    # 실제 Style 차이 계산
    actual_diff_norm = tgt_style_norm - curr_style_norm
    actual_diff_raw = actual_diff_norm * RANGE_VALS
    
    # 입력 데이터 텐서 변환 (스케일링 포함)
    curr_iq_scaled = scaler_X.transform(curr_iq_raw)
    tgt_iq_scaled = scaler_X.transform(tgt_iq_raw)
    curr_tensor = torch.tensor(curr_iq_scaled, dtype=torch.float32).to(device)
    tgt_tensor = torch.tensor(tgt_iq_scaled, dtype=torch.float32).to(device)

    # 4. 5개 그룹 모델 순차 추론 및 결과 병합
    all_pred_diff_norm = np.zeros((N, 37)) # 37개 전체 결과를 담을 빈 행렬
    
    print("\n[Starting Inference]")
    for group_name, param_indices in groups.items():
        num_params = len(param_indices)
        if num_params == 0:
            continue # 해당 그룹에 속한 파라미터가 없으면 패스
            
        model_path = os.path.join(model_dir, f'siamese_style_model_group_{group_name}.pth')
        if not os.path.exists(model_path):
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
            continue
            
        # 해당 그룹의 출력 차원(num_params)에 맞춰 모델 로드
        model = SiameseStyleRegressor(input_dim=12, output_dim=num_params).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        with torch.no_grad():
            # 그룹별 파라미터 묶음 예측
            pred_group = model(curr_tensor, tgt_tensor).cpu().numpy()
            
            # 예측된 결과들을 전체 37차원 행렬 중 알맞은 인덱스(열)에 삽입
            all_pred_diff_norm[:, param_indices] = pred_group
            
        print(f" > Group [{group_name}] Model Loaded & Inference Complete. (Output Dim: {num_params})")

    # 5. 오차 지표 계산 (물리적 스케일 복원 기준)
    pred_diff_raw = all_pred_diff_norm * RANGE_VALS
    
    abs_error = np.abs(pred_diff_raw - actual_diff_raw)
    sq_error = np.square(pred_diff_raw - actual_diff_raw)
    huber_error = np.where(abs_error <= delta, 0.5 * sq_error, delta * abs_error - 0.5 * (delta ** 2))
    
    overall_mae = np.mean(abs_error, axis=0)
    overall_mse = np.mean(sq_error, axis=0)
    overall_huber = np.mean(huber_error, axis=0)

    # 6. 엑셀 저장을 위한 데이터 구성
    style_cols = [f"Style_{i+1:02d}" for i in range(37)]
    iq_cols = [f"IQ_{i+1:02d}" for i in range(12)]
    
    rows = []
    # [요약] 전체 파라미터 요약 정보
    rows.append(["ALL_SUMMARY", "0_MAE_Original"] + [None]*12 + overall_mae.tolist())
    rows.append(["ALL_SUMMARY", "0_MSE_Original"] + [None]*12 + overall_mse.tolist())
    rows.append(["ALL_SUMMARY", f"0_Huber_Original(d={delta})"] + [None]*12 + overall_huber.tolist())
    rows.append([None] * (2 + 12 + 37))

    # [개별 데이터]
    for i in range(N):
        p_id = f"Pair_{i+1:05d}"
        rows.append([p_id, "1_Actual_Diff_Raw"] + [None]*12 + actual_diff_raw[i].tolist())
        rows.append([p_id, "2_Pred_Diff_Raw"] + [None]*12 + pred_diff_raw[i].tolist())
        rows.append([p_id, "3_Absolute_Error(MAE)"] + [None]*12 + abs_error[i].tolist())
        rows.append([p_id, "4_Squared_Error(MSE)"] + [None]*12 + sq_error[i].tolist())
        rows.append([None] * (2 + 12 + 37))

    result_df = pd.DataFrame(rows, columns=["ID", "Type"] + iq_cols + style_cols)
    result_df.to_excel(output_file_path, index=False)
    print(f"\n✅ 5개 그룹 모델 통합 테스트 완료! 결과 저장됨: {output_file_path}")

# ==========================================
# 3. 메인 실행 루프
# ==========================================
if __name__ == '__main__':
    # 설정값 (환경에 맞게 수정)
    MODEL_DIRECTORY = 'saved_models'      # 5개 그룹 모델이 들어있는 폴더
    SCALER_PATH = 'scaler_x.pkl'         # 공통 스케일러 경로
    TEST_DATA_PATH = 'data.xlsx'         # 테스트할 데이터 파일
    REF_LIMITS_PATH = 'reference.xlsx'    # Min/Max 범위가 있는 파일
    
    evaluate_grouped_models(
        model_dir=MODEL_DIRECTORY,
        scaler_path=SCALER_PATH,
        test_file_path=TEST_DATA_PATH,
        ref_file_path=REF_LIMITS_PATH,
        output_file_path="final_grouped_test_results.xlsx"
    )