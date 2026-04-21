import pandas as pd
import numpy as np
import torch

# ==========================================
# 0. 도메인 설정: 엑셀에서 Min/Max 자동 로드
# ==========================================
DOMAIN_FILE = 'domain_config.xlsx'  

try:
    df_domain = pd.read_excel(DOMAIN_FILE, header=None)
    MIN_VALS = df_domain.iloc[2, 1:38].values.astype(float)
    MAX_VALS = df_domain.iloc[3, 1:38].values.astype(float)
    RANGE_VALS = MAX_VALS - MIN_VALS
    
    print(f"✅ 도메인 설정 로드 완료: {DOMAIN_FILE}")
except Exception as e:
    print(f"❌ 도메인 파일 로드 중 에러 발생: {e}")
    MIN_VALS = np.array([0.0] * 37)
    MAX_VALS = np.array([1.0] * 37)
    RANGE_VALS = MAX_VALS - MIN_VALS

# ==========================================
# 9. 전체 데이터 쌍 종합 오차 분석 (MAE, MSE, Huber 추가)
# ==========================================
# delta 매개변수: Huber Loss에서 MAE와 MSE를 구분하는 기준값 (기본 1.0)
def export_all_pairs_comprehensive(model, scaler_X, test_file_path, output_file_path="test_results_final.xlsx", delta=1.0):
    print(f"\n--- 전체 데이터 종합 테스트 시작 (MAE, MSE, Huber Loss 평가) ---")
    
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
    
    # [개별 데이터 분석] 5000개 제한 추출
    for i in range(min(N, 5000)): 
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