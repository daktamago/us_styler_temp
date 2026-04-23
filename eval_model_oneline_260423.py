import torch
import pandas as pd
import numpy as np

def evaluate_model(model, scaler_X, test_file_path, ref_file_path, output_file_path="test_results_final.xlsx", restore=1, delta=1.0):
    print(f"\n--- 단일 입력(1:1) 데이터 종합 테스트 시작 (MAE, MSE, Huber Loss 평가) ---")

    # [수정안됨] 기준 파일에서 Min/Max 로드 (37개 출력 기준)
    df_domain = pd.read_excel(ref_file_path, header=None)
    MIN_VALS = df_domain.iloc[2, 1:38].values.astype(float)
    MAX_VALS = df_domain.iloc[3, 1:38].values.astype(float)
    RANGE_VALS = MAX_VALS - MIN_VALS    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  
    
    # 1. 데이터 로드 (Pair 구성 로직 제거)
    df_test = pd.read_excel(test_file_path)
    raw_data = df_test.values
    N = len(raw_data)
    
    # [수정됨] 입력(12)과 출력(37) 분리 (tgt_data 관련 코드 완전 삭제)
    iq_raw = raw_data[:, :12]
    style_norm = raw_data[:, 12:49] # 37차원
    
    # 2. 입력(X) 정규화 및 단일 추론
    iq_scaled = scaler_X.transform(iq_raw)
    iq_tensor = torch.tensor(iq_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # [수정됨] 모델에 단일 입력(iq_tensor)만 전달
        pred_norm = model(iq_tensor).cpu().numpy()
        
    actual_norm = style_norm
    
    # 3. 물리적 스케일 복원
    if restore == 0 :
        pred_raw = pred_norm
        actual_raw = actual_norm
    else :        
        pred_raw = pred_norm * RANGE_VALS
        actual_raw = actual_norm * RANGE_VALS
    
    # 4. ★ 3대 오차 지표 계산
    abs_error = np.abs(pred_raw - actual_raw)
    sq_error = np.square(pred_raw - actual_raw)
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
    for i in range(N): 
        # Pair 대신 Sample로 라벨링 변경
        s_id = f"Sample_{i+1:05d}"
        
        # 엑셀에 보기 편하도록 12개 원본 입력값(IQ)도 함께 기록
        current_iq = iq_raw[i].tolist()
        
        # Diff라는 명칭을 빼고 직관적인 Actual, Pred로 변경
        rows.append([s_id, "1_Actual_Raw"] + current_iq + actual_raw[i].tolist())
        rows.append([s_id, "2_Pred_Raw"] + current_iq + pred_raw[i].tolist())
        rows.append([s_id, "3_Absolute_Error(MAE)"] + [None]*12 + abs_error[i].tolist())
        rows.append([s_id, "4_Squared_Error(MSE)"] + [None]*12 + sq_error[i].tolist())
        rows.append([s_id, "5_Huber_Loss"] + [None]*12 + huber_error[i].tolist())
        rows.append([None] * (2 + 12 + 37)) # 구분선
        
    # 데이터프레임 변환 및 저장
    result_df = pd.DataFrame(rows, columns=["ID", "Type"] + iq_cols + style_cols)
    result_df.to_excel(output_file_path, index=False)
    print(f"✅ 단일 매핑 모델 종합 오차 분석 완료! 결과 저장됨: {output_file_path}")