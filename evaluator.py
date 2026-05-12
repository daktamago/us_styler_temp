import torch
import pandas as pd
import numpy as np
import os

def evaluate_model(model, scaler_X, test_file_path, ref_file_path, 
                   iq_indices, style_indices, style_names, 
                   reg_indices_local, cls_indices_local, cls_num_classes_list,
                   output_file_path="test_results.csv", restore=0, device='cuda', iq_dim=60):
    
    print(f"  [Eval] 평가 진행 중... -> {os.path.basename(output_file_path)}")
    model.eval()
    
    # 데이터 로드
    df_test = pd.read_csv(test_file_path)
    iq_cols = df_test.columns[:iq_dim].tolist() # 출력용 IQ 컬럼명 추출
    
    X_test_raw = df_test.iloc[:, :iq_dim].values
    y_test_norm = df_test.iloc[:, iq_dim:].values[:, style_indices]
    
    # IQ 파라미터 스케일링
    X_test_scaled = scaler_X.transform(X_test_raw)[:, iq_indices]
    
    # 복원용 Range 계산
    ref_df = pd.read_csv(ref_file_path, index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.capitalize()
    range_vals = np.array([float(ref_df.loc['Max', col]) - float(ref_df.loc['Min', col]) for col in style_names])

    with torch.no_grad():
        inputs = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        pred_reg, pred_cls_list = model(inputs)
        
        # 예측값 통합 (Regression + Classification)
        pred_norm = np.zeros((len(X_test_raw), len(style_names)))
        
        if len(reg_indices_local) > 0:
            pred_norm[:, reg_indices_local] = pred_reg.cpu().numpy()
            
        for i, cls_idx in enumerate(cls_indices_local):
            probs = torch.softmax(pred_cls_list[i], dim=1)
            pred_class = torch.argmax(probs, dim=1).float()
            K = cls_num_classes_list[i]
            pred_norm[:, cls_idx] = ((pred_class / (K - 1)) * 2.0 - 1.0).cpu().numpy() if K > 1 else 0.0

    # 스케일 복원 적용
    if restore == 1:
        pred_raw = pred_norm * range_vals
        actual_raw = y_test_norm * range_vals
    else:
        pred_raw = pred_norm
        actual_raw = y_test_norm

    # 오차(Absolute Error) 및 평균(MAE) 계산
    abs_error = np.abs(pred_raw - actual_raw)
    overall_mae = np.mean(abs_error, axis=0)
    
    # ---------------------------------------------------------
    # CSV 저장 (개별 데이터 상세 기록)
    # ---------------------------------------------------------
    rows = []
    
    # 1. 전체 평균 요약 정보 (최상단)
    rows.append(["ALL_SUMMARY", "0_Mean_Absolute_Error"] + [None]*iq_dim + overall_mae.tolist())
    rows.append([None] * (2 + iq_dim + len(style_names))) # 구분선
    
    # 2. 각 테스트 샘플별 결과 기록
    for i in range(len(X_test_raw)):
        p_id = f"Test_{i+1:04d}"
        iq_vals = X_test_raw[i].tolist() # 원본 IQ 파라미터 값 기록
        
        # 해당 샘플의 실제 Style 값, 예측 Style 값, 절대 오차를 순서대로 기록
        rows.append([p_id, "1_Actual_Style"] + iq_vals + actual_raw[i].tolist())
        rows.append([p_id, "2_Predicted_Style"] + iq_vals + pred_raw[i].tolist())
        rows.append([p_id, "3_Absolute_Error"] + iq_vals + abs_error[i].tolist())
        rows.append([None] * (2 + iq_dim + len(style_names))) # 샘플 간 구분선
        
    # 데이터프레임 변환 후 CSV 저장
    result_df = pd.DataFrame(rows, columns=["ID", "Type"] + iq_cols + list(style_names))
    result_df.to_csv(output_file_path, index=False)
    
    print(f"[Success] 상세 평가 결과가 저장되었습니다: {output_file_path}")