import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

# ==========================================
# 7. 학습된 모델로 새로운 테스트 데이터 평가하기
# ==========================================
def test_model_direct(model, scaler_X, scaler_y, test_file_path):
    print(f"\n--- Testing Model on: {test_file_path} ---")
    
    # 1. 테스트 데이터(Excel) 로드
    df_test = pd.read_excel(test_file_path)
    X_test_raw = df_test.iloc[:, 0:12].values
    y_test_raw = df_test.iloc[:, 12:].values
    
    # 2. 저장해둔 scaler_X, scaler_y를 사용해 테스트 데이터 정규화 (새로 fit하지 않음!)
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)
    
    # 3. 앞서 정의한 데이터셋 클래스 재사용 (동적 랜덤 페어링 적용)
    test_dataset = StyleDifferenceDataset(X_test_scaled, y_test_scaled)
    # 셔플을 끄고 순차적으로 배치 구성
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 4. 디바이스 할당 및 모델 평가 모드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 추론 시 필수
    
    all_pred_diff_scaled = []
    all_actual_diff_scaled = []
    
    # 5. 추론 시작 (기울기 계산 금지)
    with torch.no_grad():
        for curr_iq, tgt_iq, actual_diff in test_loader:
            curr_iq = curr_iq.to(device)
            tgt_iq = tgt_iq.to(device)
            
            # 모델 예측 (정규화된 차이값)
            pred_diff = model(curr_iq, tgt_iq)
            
            all_pred_diff_scaled.append(pred_diff.cpu().numpy())
            all_actual_diff_scaled.append(actual_diff.numpy())
            
    # 리스트에 담긴 배치 결과를 하나의 Numpy 배열로 병합
    pred_diff_scaled_np = np.concatenate(all_pred_diff_scaled, axis=0)
    actual_diff_scaled_np = np.concatenate(all_actual_diff_scaled, axis=0)
    
    # 6. 정규화된 예측값과 실제값을 원래 Style 스케일로 복원 (inverse_transform)
    final_pred_diff = scaler_y.inverse_transform(pred_diff_scaled_np)
    final_actual_diff = scaler_y.inverse_transform(actual_diff_scaled_np)
    
    # 7. 전체 테스트 셋에 대한 원본 스케일 MAE (평균 절대 오차) 계산
    mae = np.mean(np.abs(final_pred_diff - final_actual_diff))
    print(f"✅ Total Test MAE (Original Scale): {mae:.4f}")
    
    # 8. 샘플 3개만 뽑아서 눈으로 직접 비교해보기 (처음 5개 Style 파라미터만 출력)
    print("\n[Sample Comparison - First 5 Style Params]")
    for i in range(3):
        print(f"Sample {i+1}:")
        # 보기 편하게 소수점 4자리까지만 포맷팅
        pred_sample = np.round(final_pred_diff[i][:5], 4)
        actual_sample = np.round(final_actual_diff[i][:5], 4)
        print(f"  Predicted Difference : {pred_sample} ...")
        print(f"  Actual Difference    : {actual_sample} ...")
        print("-" * 50)

# ==========================================
# 통합 실행부 예시
# ==========================================
if __name__ == '__main__':
    # 1. 앞서 작성된 학습 코드를 돌려 메모리에 객체 할당
    # trained_model, final_scaler_X, final_scaler_y = train_model()
    
    # 2. 테스트에 사용할 엑셀 파일 경로 지정 (학습 데이터와 다른, 혹은 분리해둔 데이터 파일)
    TEST_FILE_PATH = 'test_data.xlsx' 
    
    # 3. 방금 학습된 객체들을 다이렉트로 집어넣어 테스트 진행
    # test_model_direct(trained_model, final_scaler_X, final_scaler_y, TEST_FILE_PATH)