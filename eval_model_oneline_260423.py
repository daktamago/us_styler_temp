import torch
import numpy as np

def evaluate_model(model, dataloader, device, head_dims=[12, 12, 13]):
    """
    학습된 Direct 모델을 평가하고, 전체 및 그룹(Head)별 성능을 출력합니다.
    """
    model.eval()
    criterion = torch.nn.HuberLoss() # 학습 때 사용한 Loss와 동일하게 세팅
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # 1. 단일 입력(inputs)과 정답(targets)만 받아옵니다.
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 2. 예측 및 Loss 계산
            preds = model(inputs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            
            # 3. 상세 분석을 위해 예측값과 정답을 리스트에 저장
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    # 전체 배치에 대한 평균 Loss
    avg_loss = total_loss / len(dataloader)
    
    # 누적된 예측값과 정답 배열을 하나의 Numpy 배열로 병합 (전체 데이터 개수, 37)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 전체 MAE (Mean Absolute Error, 직관적인 절대 오차) 계산
    overall_mae = np.mean(np.abs(all_preds - all_targets))
    
    print("\n" + "="*40)
    print("🎯 Model Evaluation Results")
    print("="*40)
    print(f"Overall Huber Loss : {avg_loss:.4f}")
    print(f"Overall MAE        : {overall_mae:.4f}")
    
    # ---------------------------------------------------------
    # 💡 [보너스] 멀티 헤드(그룹)별 오차(MAE) 분석
    # 각 헤드가 담당하는 구역을 잘라서 따로 평가합니다.
    # ---------------------------------------------------------
    print("\n📊 Head별 상세 오차 (MAE):")
    start_idx = 0
    for i, dim in enumerate(head_dims):
        end_idx = start_idx + dim
        
        # 각 헤드가 담당한 컬럼들만 슬라이싱
        head_preds = all_preds[:, start_idx:end_idx]
        head_targets = all_targets[:, start_idx:end_idx]
        
        head_mae = np.mean(np.abs(head_preds - head_targets))
        print(f" - Head {i+1} (파라미터 {dim}개) MAE: {head_mae:.4f}")
        
        start_idx = end_idx
        
    print("="*40 + "\n")
    
    return avg_loss, overall_mae, all_preds, all_targets