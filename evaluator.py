import torch
import pandas as pd
import numpy as np
import os

def evaluate_model(model, scaler_X, test_file_path, ref_file_path, 
                   iq_indices, style_indices, style_names, 
                   reg_indices_local, cls_indices_local, cls_num_classes_list,
                   output_file_path="test_results.csv", 
                   restore=1, delta=1.0, device='cuda', iq_dim=60):
    
    print(f"  📊 평가 진행 중... -> {os.path.basename(output_file_path)}")
    model = model.to(device)
    model.eval()  
    
    df_test = pd.read_csv(test_file_path)
    X_test_raw_full = df_test.iloc[:, :iq_dim].values
    y_test_norm_full = df_test.iloc[:, iq_dim:].values
    
    y_test_sub = y_test_norm_full[:, style_indices]
    N = len(X_test_raw_full)
    
    # 2. 복원용 Range 계산
    ref_df = pd.read_csv(ref_file_path, index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.capitalize()
    
    range_vals = np.zeros(len(style_names))
    for i, col_name in enumerate(style_names):
        if col_name in ref_df.columns:
            min_val = float(ref_df.loc['Min', col_name])
            max_val = float(ref_df.loc['Max', col_name])
            range_vals[i] = max_val - min_val if (max_val - min_val) > 0 else 1.0
        else:
            range_vals[i] = 1.0
            
    # 3. 입력(X) 정규화
    X_test_scaled_full = scaler_X.transform(X_test_raw_full)
    X_test_scaled = X_test_scaled_full[:, iq_indices]
    
    tgt_indices = np.random.permutation(N)
    
    curr_iq_scaled = X_test_scaled
    curr_style_norm = y_test_sub
    tgt_iq_scaled = X_test_scaled[tgt_indices]
    tgt_style_norm = y_test_sub[tgt_indices]
    
    curr_tensor = torch.tensor(curr_iq_scaled, dtype=torch.float32).to(device)
    tgt_tensor = torch.tensor(tgt_iq_scaled, dtype=torch.float32).to(device)
    
    # 4. 모델 추론 및 Multi-Task 결과 병합 (Reconstruction)
    with torch.no_grad():
        pred_reg, pred_cls_list = model(curr_tensor, tgt_tensor)
        
        # 병합을 담을 빈 텐서 생성
        pred_diff_norm = torch.zeros((N, len(style_indices))).to(device)
        
        # Regression 결과 채워넣기
        if len(reg_indices_local) > 0:
            pred_diff_norm[:, reg_indices_local] = pred_reg
            
        # Classification 결과(정수 인덱스 -> 연속형 -1~1 로 복원) 채워넣기
        if len(cls_indices_local) > 0:
            for i, cls_idx in enumerate(cls_indices_local):
                c = torch.argmax(pred_cls_list[i], dim=1).float()
                K = cls_num_classes_list[i]
                val = (c / (K - 1)) * 2.0 - 1.0 if K > 1 else 0.0
                pred_diff_norm[:, cls_idx] = val
                
        pred_diff_norm = pred_diff_norm.cpu().numpy()
        
    actual_diff_norm = tgt_style_norm - curr_style_norm
    
    # 5. 스케일 복원
    if restore == 0:
        pred_diff_raw, actual_diff_raw = pred_diff_norm, actual_diff_norm
    else:        
        pred_diff_raw, actual_diff_raw = pred_diff_norm * range_vals, actual_diff_norm * range_vals

    # 6. 오차 계산
    abs_error = np.abs(pred_diff_raw - actual_diff_raw)
    overall_mae = np.mean(abs_error, axis=0)
    
    # 7. CSV 저장
    iq_cols_names = [f"IQ_Input_{i+1:02d}" for i in range(len(iq_indices))]
    style_cols_names = list(style_names)
    
    rows = []
    rows.append(["ALL_SUMMARY", "0_MAE_Original"] + [None]*len(iq_cols_names) + overall_mae.tolist())
    rows.append([None] * (2 + len(iq_cols_names) + len(style_cols_names))) 
    
    for i in range(N): 
        p_id = f"Pair_{i+1:05d}"
        rows.append([p_id, "1_Actual_Diff"] + [None]*len(iq_cols_names) + actual_diff_raw[i].tolist())
        rows.append([p_id, "2_Pred_Diff"] + [None]*len(iq_cols_names) + pred_diff_raw[i].tolist())
        rows.append([p_id, "3_Absolute_Error"] + [None]*len(iq_cols_names) + abs_error[i].tolist())
        
    result_df = pd.DataFrame(rows, columns=["ID", "Type"] + iq_cols_names + style_cols_names)
    result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')