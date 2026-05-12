import torch
import pandas as pd
import numpy as np
import os

def evaluate_model(model, scaler_X, test_file_path, ref_file_path, 
                   iq_indices, style_indices, style_names, 
                   output_file_path="test_results.xlsx", 
                   restore=1, delta=1.0, device='cuda', iq_dim=60):
    
    print(f"\n--- 📊 [{os.path.basename(output_file_path)}] 모델 평가 시작 ---")
    
    model = model.to(device)
    model.eval()  
    
    # 1. 테스트 데이터 로드 및 분리
    df_test = pd.read_excel(test_file_path, engine='openpyxl')
    
    # 전체 데이터에서 IQ와 Style 분리
    X_test_raw_full = df_test.iloc[:, :iq_dim].values
    y_test_norm_full = df_test.iloc[:, iq_dim:].values
    
    # 전달받은 인덱스(학습에 사용된 차원)만 추출
    y_test_sub = y_test_norm_full[:, style_indices]
    N = len(X_test_raw_full)
    
    # 2. Min-Max Reference 로드 및 스케일 복원용 Range 계산
    # 동적 매칭: 평가 중인 Style 파라미터 이름(style_names)에 해당하는 Range만 추출
    ref_df = pd.read_excel(ref_file_path, engine='openpyxl', index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.capitalize()
    
    # 타겟 파라미터의 Max - Min 계산
    range_vals = np.zeros(len(style_names))
    for i, col_name in enumerate(style_names):
        if col_name in ref_df.columns:
            min_val = float(ref_df.loc['Min', col_name])
            max_val = float(ref_df.loc['Max', col_name])
            range_vals[i] = max_val - min_val
        else:
            range_vals[i] = 1.0 # Ref에 없으면 기본값 1.0 적용
            
    # 3. 입력(X) 정규화 및 랜덤 페어(Target-Current) 생성
    X_test_scaled_full = scaler_X.transform(X_test_raw_full)
    X_test_scaled = X_test_scaled_full[:, iq_indices]
    
    tgt_indices = np.random.permutation(N)
    
    curr_iq_scaled = X_test_scaled
    curr_style_norm = y_test_sub
    
    tgt_iq_scaled = X_test_scaled[tgt_indices]
    tgt_style_norm = y_test_sub[tgt_indices]
    
    curr_tensor = torch.tensor(curr_iq_scaled, dtype=torch.float32).to(device)
    tgt_tensor = torch.tensor(tgt_iq_scaled, dtype=torch.float32).to(device)
    
    # 4. 모델 추론
    with torch.no_grad():
        pred_diff_norm = model(curr_tensor, tgt_tensor).cpu().numpy()
        
    actual_diff_norm = tgt_style_norm - curr_style_norm
    
    # 5. 스케일 복원 (Restore)
    if restore == 0:
        pred_diff_raw = pred_diff_norm
        actual_diff_raw = actual_diff_norm
    else:        
        pred_diff_raw = pred_diff_norm #* range_vals
        actual_diff_raw = actual_diff_norm #* range_vals

    # 6. 3대 오차 지표 계산
    abs_error = np.abs(pred_diff_raw - actual_diff_raw)
    sq_error = np.square(pred_diff_raw - actual_diff_raw)
    huber_error = np.where(abs_error <= delta, 
                           0.5 * sq_error, 
                           delta * abs_error - 0.5 * (delta ** 2))
    
    overall_mae = np.mean(abs_error, axis=0)
    overall_mse = np.mean(sq_error, axis=0)
    overall_huber = np.mean(huber_error, axis=0)
    
    # 7. 엑셀 저장을 위한 데이터 구성 (가변 차원 대응)
    iq_cols_names = [f"IQ_Input_{i+1:02d}" for i in range(len(iq_indices))]
    style_cols_names = list(style_names)
    
    rows = []
    num_iq = len(iq_cols_names)
    num_style = len(style_cols_names)
    
    # [상단 요약]
    rows.append(["ALL_SUMMARY", "0_MAE_Original"] + [None]*num_iq + overall_mae.tolist())
    rows.append(["ALL_SUMMARY", "0_MSE_Original"] + [None]*num_iq + overall_mse.tolist())
    rows.append(["ALL_SUMMARY", f"0_Huber_Original(d={delta})"] + [None]*num_iq + overall_huber.tolist())
    rows.append([None] * (2 + num_iq + num_style)) 
    
    # [개별 데이터 분석]
    for i in range(N): 
        p_id = f"Pair_{i+1:05d}"
        rows.append([p_id, "1_Actual_Diff_Raw"] + [None]*num_iq + actual_diff_raw[i].tolist())
        rows.append([p_id, "2_Pred_Diff_Raw"] + [None]*num_iq + pred_diff_raw[i].tolist())
        rows.append([p_id, "3_Absolute_Error(MAE)"] + [None]*num_iq + abs_error[i].tolist())
        rows.append([p_id, "4_Squared_Error(MSE)"] + [None]*num_iq + sq_error[i].tolist())
        rows.append([p_id, "5_Huber_Loss"] + [None]*num_iq + huber_error[i].tolist())
        rows.append([None] * (2 + num_iq + num_style)) 
        
    result_df = pd.DataFrame(rows, columns=["ID", "Type"] + iq_cols_names + style_cols_names)
    result_df.to_excel(output_file_path, index=False)
    
    print(f"✅ 평가 완료! 결과 저장됨: {output_file_path}")