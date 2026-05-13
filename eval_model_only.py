import torch
import joblib
import pandas as pd
import numpy as np
import os
from model import DynamicSiameseMultiTask
from data_processing import categorize_parameters_by_step

def get_input(prompt, default_val):
    user_val = input(f"{prompt} (기본값: {default_val}): ").strip()
    return user_val if user_val else default_val

def main():
    print("="*50)
    print(" Siamese Model Evaluation Script ")
    print("="*50)

    # 1. 필수 파일 경로 입력
    model_path = get_input("1. 학습된 모델(.pth) 경로", "model_Full.pth")
    scaler_path = get_input("2. 스케일러(.pkl) 경로", "scaler_x.pkl")
    test_file = get_input("3. 평가할 새로운 데이터(CSV) 경로", "new_test_data.csv")
    ref_file = get_input("4. Min/Max Reference 파일 경로", "ParameterMinMaxStep.csv")
    
    restore = int(get_input("5. 원본 스케일 복원 여부 (0: 정규화값, 1: 원본복원)", "1"))
    output_filename = get_input("6. 결과 저장 파일명", "eval_siamese_results.csv")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 모델 및 설정 로드
    print(f"\n[Loading] 모델 파일을 분석 중입니다...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Config가 없는 구버전 모델 파일 대응
    if 'config' not in checkpoint:
        print("[Warning] 설정 정보가 없는 구버전 모델입니다. 수동 입력을 진행합니다.")
        iq_dim = int(get_input("   - IQ Parameter 개수", "80"))
        hidden_dims = [int(x.strip()) for x in get_input("   - Encoder 차원", "128,256,512").split(',')]
        extractor_dims = [int(x.strip()) for x in get_input("   - Extractor 차원", "512,256").split(',')]
        reg_head_dims = [int(x.strip()) for x in get_input("   - Regressor Head 차원", "512,128").split(',')]
        cls_head_dims = [int(x.strip()) for x in get_input("   - Classifier Head 차원", "256").split(',')]
        
        # 임시 Config 생성
        df_temp = pd.read_csv(test_file, nrows=0)
        style_cols = df_temp.columns[iq_dim:].tolist()
        reg_idx, cls_idx, num_cls = categorize_parameters_by_step(ref_file, style_cols, 10)
        
        conf = {
            'input_dim': iq_dim, 'hidden_dims': hidden_dims, 'extractor_dims': extractor_dims,
            'reg_head_dims': reg_head_dims, 'cls_head_dims': cls_head_dims,
            'reg_dim': len(reg_idx), 'cls_num_classes_list': num_cls
        }
        state_dict = checkpoint
    else:
        conf = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
        print(f"   - 구조 감지 완료: Encoder{conf['hidden_dims']}, Extractor{conf['extractor_dims']}")

    # 3. 모델 초기화
    model = DynamicSiameseMultiTask(
        input_dim=conf['input_dim'], hidden_dims=conf['hidden_dims'], extractor_dims=conf['extractor_dims'],
        reg_head_dims=conf['reg_head_dims'], cls_head_dims=conf['cls_head_dims'],
        reg_dim=conf['reg_dim'], cls_num_classes_list=conf['cls_num_classes_list']
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. 데이터 로드 및 전처리
    df_test = pd.read_csv(test_file)
    iq_dim = conf['input_dim']
    iq_cols = df_test.columns[:iq_dim].tolist()
    style_cols = df_test.columns[iq_dim:].tolist()
    
    X_test_raw = df_test.iloc[:, :iq_dim].values
    y_test_norm = df_test.iloc[:, iq_dim:].values
    
    scaler_X = joblib.load(scaler_path)
    X_test_scaled = scaler_X.transform(X_test_raw)

    # 복원용 Range 계산 (공백 방어 로직 적용)
    ref_df = pd.read_csv(ref_file, index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.upper()
    
    # Reference 파일의 컬럼명 맵핑 딕셔너리 생성
    ref_col_map = {str(c).strip().upper(): c for c in ref_df.columns}
    
    range_vals = []
    for col in style_cols:
        col_clean = str(col).strip().upper()
        if col_clean in ref_col_map:
            ref_col_name = ref_col_map[col_clean]
            max_val = float(ref_df.loc['MAX', ref_col_name])
            min_val = float(ref_df.loc['MIN', ref_col_name])
            range_vals.append(max_val - min_val)
        else:
            # 万일 매칭 실패 시 에러 대신 경고를 띄우고 스케일 유지(1.0)
            print(f"   [Warning] '{col}' 파라미터를 Reference 파일에서 찾을 수 없습니다.")
            range_vals.append(1.0)
            
    range_vals = np.array(range_vals)

    # 파라미터 분류 인덱스 추출
    reg_idx, cls_idx, num_cls = categorize_parameters_by_step(ref_file, style_cols, 10)

    # 5. 샴 네트워크를 위한 무작위 비교군(Target) 생성
    np.random.seed(42)
    num_samples = len(X_test_raw)
    target_indices = np.random.randint(0, num_samples, size=num_samples)
    
    target_scaled = X_test_scaled[target_indices]
    actual_diff_norm = y_test_norm[target_indices] - y_test_norm

    # 6. 추론 시작
    print(f"\n[Eval] Siamese 모델 추론을 시작합니다...")
    with torch.no_grad():
        curr_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        tgt_tensor = torch.tensor(target_scaled, dtype=torch.float32).to(device)
        
        pred_reg, pred_cls_list = model(curr_tensor, tgt_tensor)
        
        pred_diff_norm = np.zeros((num_samples, len(style_cols)))
        
        if len(reg_idx) > 0:
            pred_diff_norm[:, reg_idx] = pred_reg.cpu().numpy()
            
        for i, c_idx in enumerate(cls_idx):
            probs = torch.softmax(pred_cls_list[i], dim=1)
            pred_class = torch.argmax(probs, dim=1).float()
            K = num_cls[i]
            # Classification으로 푼 값을 다시 -1.0 ~ 1.0 범위의 차이값으로 변환
            pred_diff_norm[:, c_idx] = ((pred_class / (K - 1)) * 2.0 - 1.0).cpu().numpy() if K > 1 else 0.0

    # 스케일 복원
    if restore == 1:
        pred_diff_final = pred_diff_norm * range_vals
        actual_diff_final = actual_diff_norm * range_vals
    else:
        pred_diff_final, actual_diff_final = pred_diff_norm, actual_diff_norm

    abs_error = np.abs(pred_diff_final - actual_diff_final)
    overall_mae = np.mean(abs_error, axis=0)

    # 7. CSV 작성
    rows = []
    rows.append(["ALL_SUMMARY", "0_Mean_Absolute_Error"] + [None]*iq_dim + overall_mae.tolist())
    rows.append([None] * (2 + iq_dim + len(style_cols)))

    for i in range(num_samples):
        # 샴 네트워크의 특성을 살려 어떤 데이터와 비교했는지 표기
        p_id = f"Sample_{i:04d}_vs_{target_indices[i]:04d}"
        iq_vals = X_test_raw[i].tolist() # 현재 IQ만 표기
        
        rows.append([p_id, "1_Actual_Diff"] + iq_vals + actual_diff_final[i].tolist())
        rows.append([p_id, "2_Pred_Diff"] + iq_vals + pred_diff_final[i].tolist())
        rows.append([p_id, "3_Absolute_Error"] + iq_vals + abs_error[i].tolist())
        rows.append([None] * (2 + iq_dim + len(style_cols)))

    pd.DataFrame(rows, columns=["ID(Curr_vs_Tgt)", "Type"] + iq_cols + style_cols).to_csv(output_filename, index=False)
    print(f"[Success] 평가 완료 및 저장됨: {output_filename}")

if __name__ == "__main__":
    main()