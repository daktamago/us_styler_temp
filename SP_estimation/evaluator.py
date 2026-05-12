import torch
import pandas as pd
import numpy as np
import os

def evaluate_model(model, scaler_X, test_file_path, ref_file_path, 
                   iq_indices, style_indices, style_names, 
                   reg_indices_local, cls_indices_local, cls_num_classes_list,
                   output_file_path="test_results.csv", restore=0, device='cuda', iq_dim=60):
    
    model.eval()
    df_test = pd.read_csv(test_file_path)
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

    # 스케일 복원 및 오차 계산
    if restore == 1:
        pred_raw, actual_raw = pred_norm * range_vals, y_test_norm * range_vals
    else:
        pred_raw, actual_raw = pred_norm, y_test_norm

    mae = np.mean(np.abs(pred_raw - actual_raw), axis=0)
    
    # CSV 저장 (요약본)
    result_df = pd.DataFrame([mae], columns=style_names)
    result_df.to_csv(output_file_path, index=False)
    print(f"[Success] Evaluation results saved to {output_file_path}")