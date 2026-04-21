import pandas as pd
import numpy as np
import torch

# ==========================================
# 0. 도메인 설정: 엑셀에서 Min/Max 자동 로드
# ==========================================
DOMAIN_FILE = 'domain_config.xlsx'  # 사용자가 수동으로 입력하는 파일명

try:
    # header=None으로 읽어야 Row 3, 4 위치를 정확히 인덱스로 잡을 수 있습니다.
    df_domain = pd.read_excel(DOMAIN_FILE, header=None)
    
    # Pandas 인덱스는 0부터 시작하므로:
    # Row 3 -> index 2 / Row 4 -> index 3
    # Column 2 -> index 1 / Column 38 -> index 37 (1:38 슬라이싱은 1~37까지 포함)
    MIN_VALS = df_domain.iloc[2, 1:38].values.astype(float)
    MAX_VALS = df_domain.iloc[3, 1:38].values.astype(float)
    RANGE_VALS = MAX_VALS - MIN_VALS
    
    print(f"✅ 도메인 설정 로드 완료: {DOMAIN_FILE}")
    print(f"   - 파라미터 개수: {len(MIN_VALS)}개")
except Exception as e:
    print(f"❌ 도메인 파일 로드 중 에러 발생: {e}")
    # 파일이 없을 경우를 대비한 기본값 (에러 방지용)
    MIN_VALS = np.array([0.0] * 37)
    MAX_VALS = np.array([1.0] * 37)
    RANGE_VALS = MAX_VALS - MIN_VALS

# ==========================================
# 9. 전체 데이터 쌍 종합 오차 분석 (Manual Min/Max Recovery)
# ==========================================
def export_all_pairs_comprehensive(model, scaler_X, test_file_path, output_file_path="test_results_final.xlsx"):
    print(f"\n--- 전체 데이터 종합 테스트 시작 (Min/Max 복원 모드) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  
    
    # 1. 데이터 로드
    df_test = pd.read_excel(test_file_path)
    raw_data = df_test.values
    N = len(raw_data)
    
    # 2. 랜덤 페어링 (1:1 매칭)
    tgt_indices = np.random.permutation(N)
    tgt_data = raw_data[tgt_indices]
    
    # 3. 입력(X: 12개)과 출력(y: 37개) 분리
    # X는 StandardScaler(scaler_X) 적용 대상, y는 이미 정규화된 상태
    curr_iq_raw = raw_data[:, :12]
    curr_style_norm = raw_data[:, 12:]
    
    tgt_iq_raw = tgt_data[:, :12]
    tgt_style_norm = tgt_data[:, 12:]
    
    # 4. 입력 데이터(X) 정규화
    curr_iq_scaled = scaler_X.transform(curr_iq_raw)
    tgt_iq_scaled = scaler_X.transform(tgt_iq_raw)
    
    # 5. 추론 시작
    curr_tensor = torch.tensor(curr_iq_scaled, dtype=torch.float32).to(device)
    tgt_tensor = torch.tensor(tgt_iq_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # 모델은 정규화된 차이값(Delta_norm)을 예측함
        pred_diff_norm = model(curr_tensor, tgt_tensor).cpu().numpy()
        
    # 실제 정규화된 차이값 계산
    actual_diff_norm = tgt_style_norm - curr_style_norm
    
    # 6. ★ 핵심: 원래 스케일로 복원 (Denormalization)
    # 공식: 실제 차이 = 정규화된 차이 * (Max - Min)
    pred_diff_raw = pred_diff_norm * RANGE_VALS
    actual_diff_raw = actual_diff_norm * RANGE_VALS
    
    # 7. 오차 지표 계산 (복원된 스케일 기준)
    abs_error = np.abs(pred_diff_raw - actual_diff_raw)
    overall_mae_per_param = np.mean(abs_error, axis=0)
    
    # 8. 엑셀 저장을 위한 데이터 구성
    # 37개 패러미터에 맞춰 컬럼명 생성 (예: Style_01 ~ Style_37)
    style_cols = [f"Style_{i+1:02d}" for i in range(37)]
    iq_cols = [f"IQ_{i+1:02d}" for i in range(12)]
    
    rows = []
    # 최상단 전체 요약 행 (MAE)
    summary_row = ["ALL_SUMMARY", "0_MAE_Original_Scale"] + [None]*12 + overall_mae_per_param.tolist()
    rows.append(summary_row)
    rows.append([None] * (2 + 12 + 37)) # 빈 줄
    
    for i in range(min(N, 5000)): # 너무 많으면 상위 5000개만 추출 추천
        p_id = f"Pair_{i+1:05d}"
        rows.append([p_id, "1_Actual_Diff_Raw"] + [None]*12 + actual_diff_raw[i].tolist())
        rows.append([p_id, "2_Pred_Diff_Raw"] + [None]*12 + pred_diff_raw[i].tolist())
        rows.append([p_id, "3_Absolute_Error"] + [None]*12 + abs_error[i].tolist())
        rows.append([None] * (2 + 12 + 37)) # 구분선
        
    # 데이터프레임 변환 및 저장
    result_df = pd.DataFrame(rows, columns=["ID", "Type"] + iq_cols + style_cols)
    result_df.to_excel(output_file_path, index=False)
    print(f"✅ 분석 완료! 결과 저장됨: {output_file_path}")