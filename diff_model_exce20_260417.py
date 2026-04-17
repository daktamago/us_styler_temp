import pandas as pd
import numpy as np
import torch

# ==========================================
# 8. 20개 샘플 추출 및 4행 구조(Row) 엑셀 내보내기
# ==========================================
def export_20_pairs_custom_format(model, scaler_X, scaler_y, test_file_path, output_file_path="test_results_20_custom.xlsx"):
    print(f"\n--- 20개 데이터 쌍 추출 및 엑셀 저장 시작 ---")
    
    # 1. 디바이스 할당 및 모델 평가 모드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  
    
    # 2. 테스트 데이터 로드
    df_test = pd.read_excel(test_file_path)
    
    # 3. 20개의 쌍을 만들기 위해 20개의 Current, Target 인덱스 무작위 추출
    # (결과 재현을 원하시면 np.random.seed(42) 등을 추가하셔도 좋습니다)
    total_samples = len(df_test)
    curr_indices = np.random.choice(total_samples, 20, replace=False)
    tgt_indices = np.random.choice(total_samples, 20, replace=False)
    
    rows = []
    
    # 4. 컬럼명 동적 생성 (IQ 12개, Style 나머지 전체)
    iq_dim = 12
    style_dim = df_test.shape[1] - iq_dim
    
    iq_cols = [f"IQ_{i+1}" for i in range(iq_dim)]
    style_cols = [f"Style_{i+1}" for i in range(style_dim)]
    columns = ["Pair_ID", "Data_Type"] + iq_cols + style_cols
    
    # 5. 추론 및 데이터 재구성 (기울기 계산 금지)
    with torch.no_grad():
        for i in range(20):
            curr_idx = curr_indices[i]
            tgt_idx = tgt_indices[i]
            
            # 원본 데이터 추출
            curr_data = df_test.iloc[curr_idx].values
            tgt_data = df_test.iloc[tgt_idx].values
            
            curr_iq_raw = curr_data[:12]
            curr_style_raw = curr_data[12:]
            
            tgt_iq_raw = tgt_data[:12]
            tgt_style_raw = tgt_data[12:]
            
            # (1) 실제 차이값 (Actual Difference) 계산
            actual_diff_raw = tgt_style_raw - curr_style_raw
            
            # (2) 모델 예측을 위한 IQ 정규화 (스케일러 사용)
            curr_iq_scaled = scaler_X.transform([curr_iq_raw])
            tgt_iq_scaled = scaler_X.transform([tgt_iq_raw])
            
            # 텐서 변환
            curr_iq_tensor = torch.tensor(curr_iq_scaled, dtype=torch.float32).to(device)
            tgt_iq_tensor = torch.tensor(tgt_iq_scaled, dtype=torch.float32).to(device)
            
            # (3) 모델 예측 및 차이값 스케일 복원 (평균을 더하지 않고 표준편차만 곱함!)
            pred_diff_scaled = model(curr_iq_tensor, tgt_iq_tensor).cpu().numpy()[0]
            pred_diff_raw = pred_diff_scaled * scaler_y.scale_
            
            # (4) 엑셀에 기록할 4개의 행(Row) 생성
            pair_id = f"Pair_{i+1:02d}"
            blank_iq = [None] * iq_dim  # 차이값 행에서는 IQ 부분을 비워둠(빈칸)
            
            # 1행: Current Data (IQ + Style)
            row1 = [pair_id, "1_Current"] + curr_iq_raw.tolist() + curr_style_raw.tolist()
            # 2행: Target Data (IQ + Style)
            row2 = [pair_id, "2_Target"] + tgt_iq_raw.tolist() + tgt_style_raw.tolist()
            # 3행: 실제 정답 차이 (Actual Difference)
            row3 = [pair_id, "3_Actual_Diff"] + blank_iq + actual_diff_raw.tolist()
            # 4행: 모델 예측 차이 (Predicted Difference)
            row4 = [pair_id, "4_Pred_Diff"] + blank_iq + pred_diff_raw.tolist()
            
            # 리스트에 4개 행 순차적으로 추가
            rows.extend([row1, row2, row3, row4])
            
    # 6. DataFrame 변환 및 엑셀 저장
    result_df = pd.DataFrame(rows, columns=columns)
    result_df.to_excel(output_file_path, index=False)
    print(f"✅ 20개 데이터 쌍(총 80행) 비교 결과가 '{output_file_path}'에 성공적으로 저장되었습니다.")

# ==========================================
# 통합 실행부 예시
# ==========================================
if __name__ == '__main__':
    # 기존에 학습하거나 로드한 객체들이 있다고 가정
    # TEST_FILE_PATH = 'IQ_Target_Testdata.xlsx'
    
    # 함수 실행
    # export_20_pairs_custom_format(trained_model, final_scaler_X, final_scaler_y, TEST_FILE_PATH)
    pass