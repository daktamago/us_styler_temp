# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from dataset import prepare_dataloaders
from model import TabularStyleTransformer

# =====================================================================
# 🛠️ USER CONFIGURATION (사용자 설정 영역)
# =====================================================================
CONFIG = {
    # 1. 파일 및 저장 경로
    'CSV_FILE_PATH': 'clean_data.csv',
    'SAVE_DIR': './artifacts',
    
    # 2. 데이터 구조 파라미터 (하드코딩 제거)
    'IQ_PARAM_COUNT': 60,       
    'STYLE_PARAM_COUNT': 37,    
    'IQ_GROUP_SIZE': 4,         
    
    # 3. 모델 라우팅 및 출력 파라미터
    'CLASSIFIER_NAMES': ['LimitationType', 'DirPosition-Lv2', 'DirPosition-Lv3'], 
    'MAX_CLASSES': 21,          # 분류기 출력 가짓수 (음수~양수 shift 대응)
    
    # 4. 트랜스포머 아키텍처 파라미터
    'D_MODEL': 128,             
    'N_HEADS': 4,               
    'NUM_LAYERS': 2,            
    
    # 5. 학습(Training) 하이퍼파라미터
    'NUM_PAIRS_PER_EPOCH': 100000, 
    'BATCH_SIZE': 256,
    'EPOCHS': 5,
    'LEARNING_RATE': 1e-3
}
# =====================================================================


@torch.no_grad()
def run_final_evaluation(model, val_loader, class_idx, reg_idx, style_names, config, device):
    """
    학습이 완료된 모델을 검증 데이터셋 전체에 대해 평가하고,
    지표별 요약 파일 및 샘플별 상세 정답/예측 파일 두 가지를 저장합니다.
    """
    print("🔎 검증 데이터셋 기반 최종 정밀 평가 및 아티팩트 저장을 시작합니다...")
    model.eval()
    
    all_actuals = []
    all_predictions = []
    
    shift_value = config['MAX_CLASSES'] // 2
    
    # 1. 검증 데이터 전체 추론 및 수집
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        pred_class, pred_reg, _ = model(inputs)
        
        # CPU로 이동하여 넘파이 변환 준비
        pred_class = pred_class.detach().cpu()
        pred_reg = pred_reg.detach().cpu()
        targets = targets.cpu()
        
        # 분류기 요원의 가중치 확률(Logit) 중 가장 높은 인덱스를 뽑고, 원래 음수/양수 범위로 역-Shift 보정
        pred_class_idx = pred_class.argmax(dim=-1) # [Batch, Class_개수]
        pred_class_final = pred_class_idx - shift_value
        
        # 37개 전체 파라미터 형태로 다시 조합하기 위해 빈 템플릿 생성
        batch_size = inputs.size(0)
        full_pred = np.zeros((batch_size, config['STYLE_PARAM_COUNT']), dtype=np.float32)
        
        # 동적 인덱스 맵에 맞춰 배치별 결과 삽입
        full_pred[:, class_idx] = pred_class_final.numpy()
        full_pred[:, reg_idx] = pred_reg.numpy()
        
        all_actuals.append(targets.numpy())
        all_predictions.append(full_pred)
        
    # 데이터 병합 [Total_Val_Samples, 37]
    actuals = np.concatenate(all_actuals, axis=0)
    preds = np.concatenate(all_predictions, axis=0)
    
    # 2. 파라미터 개별 지표 요약본 계산 (MAE, MSE, Huber, Accuracy)
    summary_rows = []
    huber_criterion = nn.HuberLoss(reduction='none')
    
    for i, col_name in enumerate(style_names):
        y_true = actuals[:, i]
        y_pred = preds[:, i]
        
        abs_errors = np.abs(y_true - y_pred)
        mae = np.mean(abs_errors)
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Huber Loss 계산
        huber = huber_criterion(torch.tensor(y_pred), torch.tensor(y_true)).mean().item()
        
        # 분류기 영역인 경우 Accuracy(정확도) 추가 계산, 회귀 영역이면 N/A 처리
        if i in class_idx:
            # 반올림 매칭을 통해 완벽히 일치하는 정수형 계단 일치율 확인
            accuracy = np.mean(np.round(y_true) == np.round(y_pred)) * 100
            task_type = "Classification"
        else:
            accuracy = "N/A (Regression)"
            task_type = "Regression"
            
        summary_rows.append({
            "Parameter_Name": col_name,
            "Task_Type": task_type,
            "MAE(Absolute_Error)": round(mae, 6),
            "MSE(Squared_Error)": round(mse, 6),
            "Huber_Loss": round(huber, 6),
            "Accuracy_Percent(Only Class)": accuracy
        })
        
    # 요약본 데이터프레임 저장
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(config['SAVE_DIR'], 'evaluation_summary.csv'), index=False)
    print(f"📊 요약 성능 지표 리포트 저장 완료 -> {config['SAVE_DIR']}/evaluation_summary.csv")
    
    # 3. 샘플별 실제 정답 vs 예측값 대조 상세본 파일 생성 (최대 5000개 샘플 추출 보존)
    detailed_rows = []
    num_samples_to_save = min(5000, len(actuals))
    
    for sample_idx in range(num_samples_to_save):
        for param_idx, col_name in enumerate(style_names):
            detailed_rows.append({
                "Sample_ID": f"Pair_Sample_{sample_idx:05d}",
                "Parameter_Name": col_name,
                "Actual_Difference": actuals[sample_idx, param_idx],
                "Predicted_Difference": preds[sample_idx, param_idx],
                "Absolute_Error": abs(actuals[sample_idx, param_idx] - preds[sample_idx, param_idx])
            })
            
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(os.path.join(config['SAVE_DIR'], 'evaluation_detailed_pairs.csv'), index=False)
    print(f"📋 샘플 대조 상세 리포트 저장 완료 -> {config['SAVE_DIR']}/evaluation_detailed_pairs.csv")


def train_and_observe(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 디바이스 환경: {device}")
    
    # 💡 수정됨: 데이터로더에서 scaler도 함께 받아옵니다.
    train_loader, val_loader, class_idx, reg_idx, style_names, scaler = prepare_dataloaders(config['CSV_FILE_PATH'], config)
    
    num_iq_groups = config['IQ_PARAM_COUNT'] // config['IQ_GROUP_SIZE']
    iq_group_names = [f"IQ_Group_{i}" for i in range(num_iq_groups)]
    
    model = TabularStyleTransformer(class_idx, reg_idx, config).to(device)
    
    criterion_reg = nn.HuberLoss() 
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-3)
    
    # --- [학습 루프 구동 구역] ---
    for epoch in range(config['EPOCHS']):
        model.train()
        total_loss = 0
        all_attn_weights = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            pred_class, pred_reg, attn_weights = model(inputs)
            
            target_class_raw = targets[:, class_idx].long()
            target_reg = targets[:, reg_idx]
            
            shift_value = config['MAX_CLASSES'] // 2
            target_class_shifted = target_class_raw + shift_value
            target_class_shifted = torch.clamp(target_class_shifted, 0, config['MAX_CLASSES'] - 1)
            
            loss_class = criterion_class(pred_class.transpose(1, 2), target_class_shifted)
            loss_reg = criterion_reg(pred_reg, target_reg)
            
            loss = loss_reg + (loss_class * 0.5) 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_attn_weights.append(attn_weights.detach().cpu())
            
        print(f"Epoch [{epoch+1}/{config['EPOCHS']}] - Avg Train Loss: {total_loss / len(train_loader):.4f}")
        
        # 매 에폭 종료 시 어텐션 행렬 기록
        epoch_mean_attn = torch.cat(all_attn_weights, dim=0).mean(dim=0).numpy()
        df_attn = pd.DataFrame(epoch_mean_attn, index=style_names, columns=iq_group_names)
        df_attn.to_csv(os.path.join(config['SAVE_DIR'], f'cross_attention_matrix_epoch_{epoch+1}.csv'))
        
    # --- [학습 완료 후 프로세스 실행 구역] ---
    # 1. 가중치 기반 설명 가능성 파일들 추출
    model.save_explainable_artifacts(style_names, save_dir=config['SAVE_DIR'])
    
    run_final_evaluation(model, val_loader, class_idx, reg_idx, style_names, config, device)
    
    torch.save(model.state_dict(), os.path.join(config['SAVE_DIR'], "best_style_transformer.pth"))
    
    # 💡 추가됨: 학습 완료 후 Scaler를 Pickle 파일로 영구 저장 (추론 시 재사용 목적)
    scaler_path = os.path.join(config['SAVE_DIR'], 'iq_minmax_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"💾 [스케일러 저장 완료] 향후 추론 시 {scaler_path} 를 로드하여 사용하세요.")
    
    print("🎯 [성능 평가 완료] 전체 프로세스가 성공적으로 마감되었습니다.")

if __name__ == "__main__":
    if os.path.exists(CONFIG['CSV_FILE_PATH']):
        train_and_observe(CONFIG)
    else:
        print(f"💡 {CONFIG['CSV_FILE_PATH']} 파일 매칭에 실패했습니다.")