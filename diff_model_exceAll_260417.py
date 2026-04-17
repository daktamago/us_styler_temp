import pandas as pd
import numpy as np
import torch

# ==========================================
# 9. 전체 데이터 쌍 테스트 및 종합 오차 분석 엑셀 내보내기
# ==========================================
def export_all_pairs_comprehensive(model, scaler_X, scaler_y, test_file_path, output_file_path="test_results_all_comprehensive.xlsx"):
    print(f"\n--- 전체 데이터 종합 테스트 및 엑셀 저장 시작 ---")
    
    # 1. 디바이스 할당 및 모델 평가 모드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  
    
    # 2. 전체 테스트 데이터 로드
    df_test = pd.read_excel(test_file_path)
    curr_data = df_test.values
    N = len(curr_data)
    
    # 3. 전체 데이터를 한 번씩 모두 Pair로 만들기 (한 칸씩 밀어서 꼬리잡기 방식으로 매칭)
    tgt_data = np.roll(curr_data, shift=-1, axis=0) 
    
    # 4. 데이터 분리
    curr_iq_raw = curr_data[:, :12]
    curr_style_raw = curr_data[:, 12:]
    tgt_iq_raw = tgt_data[:, :12]
    tgt_style_raw = tgt_data[:, 12:]
    
    # 실제 차이값 계산
    actual_diff_raw = tgt_style_raw - curr_style_raw
    
    # 5. 모델 예측을 위한 전체 입력 데이터 스케일링 (한 번에 Batch 처리)
    curr_iq_scaled = scaler_X.transform(curr_iq_raw)
    tgt_iq_scaled = scaler_X.transform(tgt_iq_raw)
    
    # 텐서 변환
    curr_iq_tensor = torch.tensor(curr_iq_scaled, dtype=torch.float32).to(device)
    tgt_iq_tensor = torch.tensor(tgt_iq_scaled, dtype=torch.float32).to(device)
    
    # 6. 전체 데이터 추론 (기울기 계산 금지)
    with torch.no_grad():
        pred_diff_scaled = model(curr_iq_tensor, tgt_iq_tensor).cpu().numpy()
        
    # 모델 예측 차이값 스케일 복원 (평균을 더하지 않고 표준편차만 곱함!)
    pred_diff_raw = pred_diff_scaled * scaler_y.scale_
    
    # ==========================================
    # 📊 오차 지표(Metrics) 계산
    # ==========================================
    # (1) 요소별 절대 오차 (Absolute Error)
    abs_error_raw = np.abs(actual_diff_raw - pred_diff_raw)
    
    # (2) 각 Pair별 종합 MAE (행 단위 평균)
    pair_mae = np.mean(abs_error_raw, axis=1)
    
    # (3) ★ 각 Style 파라미터별 전체 종합 MAE (열 단위 평균)
    overall_style_mae = np.mean(abs_error_raw, axis=0)
    
    # ==========================================
    # 📝 엑셀 기록용 데이터 구조화
    # ==========================================
    iq_dim = 12
    style_dim = curr_style_raw.shape[1]
    
    iq_cols = [f"IQ_{i+1}" for i in range(iq_dim)]
    style_cols = [f"Style_{i+1}" for i in range(style_dim)]
    columns = ["Pair_ID", "Data_Type"] + iq_cols + style_cols
    
    rows = []
    
    # ★ 추가 사항 2: 가장 윗 줄에 각 Style 파라미터별 전체 종합 오차(MAE) 기록
    top_row = ["ALL_PAIRS_SUMMARY", "0_Overall_Style_MAE"] + [None] * iq_dim + overall_style_mae.tolist()
    rows.append(top_row)
    rows.append(["", ""] + [None] * (iq_dim + style_dim)) # 시각적 구분을 위한 빈 줄
    
    # 각 Pair별 5행 데이터 기록
    for i in range(N):
        pair_id = f"Pair_{i+1:05d}"
        blank_iq = [None] * iq_dim
        
        # 1행: Current Data
        row1 = [pair_id, "1_Current"] + curr_iq_raw[i].tolist() + curr_style_raw[i].tolist()
        # 2행: Target Data
        row2 = [pair_id, "2_Target"] + tgt_iq_raw[i].tolist() + tgt_style_raw[i].tolist()
        # 3행: 실제 정답 차이
        row3 = [pair_id, "3_Actual_Diff"] + blank_iq + actual_diff_raw[i].tolist()
        # 4행: 모델 예측 차이
        row4 = [pair_id, "4_Pred_Diff"] + blank_iq + pred_diff_raw[i].tolist()
        # ★ 추가 사항 1: 5행에 절대 오차 기록 (라벨에 해당 Pair의 MAE 명시)
        row5 = [pair_id, f"5_Abs_Error (Pair_MAE: {pair_mae[i]:.3f})"] + blank_iq + abs_error_raw[i].tolist()
        
        rows.extend([row1, row2, row3, row4, row5])
            
    # DataFrame 변환 및 엑셀 저장
    result_df = pd.DataFrame(rows, columns=columns)
    result_df.to_excel(output_file_path, index=False)
    print(f"✅ 전체 데이터 쌍(총 {N}개 Pair) 종합 분석 결과가 '{output_file_path}'에 성공적으로 저장되었습니다.")

# ==========================================
# 실행부 예시
# ==========================================
if __name__ == '__main__':
    # TEST_FILE_PATH = 'IQ_Target_Testdata.xlsx'
    # export_all_pairs_comprehensive(model, loaded_scaler_X, loaded_scaler_y, TEST_FILE_PATH)
    pass