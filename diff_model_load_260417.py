import torch
import joblib
import numpy as np
# SiameseStyleRegressor 클래스 코드가 이 파일에도 동일하게 정의되어 있어야 합니다.

# 1. 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 스케일러 복원 (입력용, 출력용 모두 로드)
loaded_scaler_X = joblib.load('scaler_x.pkl')
loaded_scaler_y = joblib.load('scaler_y.pkl')

# 3. 모델 아키텍처 인스턴스화 및 가중치 복원
model = SiameseStyleRegressor(input_dim=12, hidden_dim=64, output_dim=71).to(device)
model.load_state_dict(torch.load('siamese_style_model.pth', map_location=device))

# 4. 평가 모드로 전환 (Dropout, BatchNorm 고정 등 추론 시 필수)
model.eval()

print("모델 및 스케일러 로드 완료. 추론 준비 끝.")

# ==========================================
# 실제 추론 시뮬레이션
# ==========================================

# 가상의 새로운 Raw 데이터 (실제 서비스에서는 엑셀/사용자 입력에서 받아온 12개 값)
# 형태: (Batch_size, 12)
raw_current_iq = np.array([[1.2, 3.4, 2.1, 0.5, 4.2, 1.1, 2.2, 3.1, 0.9, 1.5, 2.4, 3.3]])
raw_target_iq = np.array([[2.0, 3.0, 2.5, 0.8, 4.0, 1.5, 2.0, 3.5, 1.2, 1.0, 2.8, 3.0]])

# [과정 1] 입력 데이터 정규화: 반드시 저장해둔 scaler_X를 사용해 transform 진행
scaled_current_iq = loaded_scaler_X.transform(raw_current_iq)
scaled_target_iq = loaded_scaler_X.transform(raw_target_iq)

# [과정 2] 텐서로 변환 후 디바이스 할당
tensor_curr_iq = torch.tensor(scaled_current_iq, dtype=torch.float32).to(device)
tensor_tgt_iq = torch.tensor(scaled_target_iq, dtype=torch.float32).to(device)

# [과정 3] 모델 추론 (역전파 금지)
with torch.no_grad():
    pred_diff_scaled = model(tensor_curr_iq, tensor_tgt_iq)

# [과정 4] 예측된 차이값 텐서를 Numpy 배열로 변환
pred_diff_scaled_np = pred_diff_scaled.cpu().numpy()

# [과정 5] 출력 데이터 원본 스케일 복원: scaler_y를 사용해 inverse_transform 진행
final_style_diff = loaded_scaler_y.inverse_transform(pred_diff_scaled_np)

print("\n--- 최종 산출된 Style Difference (원본 스케일) ---")
print(final_style_diff)