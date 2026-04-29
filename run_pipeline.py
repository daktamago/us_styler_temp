import torch
import joblib
import os

from data_processing import prepare_raw_data, load_scale_and_group_data, categorize_parameters_by_step
from models import SiameseStyleMultiTask_Base
from trainer import run_multitask_training_pipeline
from evaluator import evaluate_model

# ==========================================
# ⚙️ 1. 설정 (Configuration)
# ==========================================
# DATA_MODE: 1 = 원본(Raw)에서 새로 전처리 후 진행 / 2 = 이미 정제된 데이터(Train/Test)로 진행
DATA_MODE = 2  

RAW_DATA_FILE = "IQ_Style_Data.csv"         # 모드 1일 때 읽을 원본
REF_FILE = "ParameterMinMaxStep_250613.csv" # MinMaxStep 파일 (경로 맞춰주세요)

TRAIN_FILE = "IQ_Style_Train.csv"
TEST_FILE = "IQ_Style_Test.csv"
SCALER_PATH = 'scaler_x.pkl'

# 하이퍼파라미터
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
STEP_THRESHOLD = 10 # 이 수치 이하는 Classification 으로 빠짐
IQ_DIM = 60

# ==========================================
# 🚀 2. 마스터 실행 함수
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== 통합 파이프라인 가동 (Device: {device}) ===")
    
    if DATA_MODE == 1:
        # 추출해야할 전체 37개 타겟 이름 리스트 (필요시 채워넣으세요)
        TARGET_COLUMNS = ["Edge Threshold-Lv2", "Edge Threshold-Lv3", "LapSmoothRate-Lv2"] 
        prepare_raw_data(RAW_DATA_FILE, REF_FILE, TARGET_COLUMNS, TRAIN_FILE, TEST_FILE, test_size=0.1, iq_dim=IQ_DIM)

    # 학습용 데이터 로드
    (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_X, 
     iq_groups, groups, style_cols, total_iq_dim, total_style_dim) = load_scale_and_group_data(TRAIN_FILE, iq_dim=IQ_DIM)
    
    joblib.dump(scaler_X, SCALER_PATH)
    
    all_iq_indices = list(range(total_iq_dim))
    all_style_indices = list(range(total_style_dim))

    # [공통 학습/평가 실행 함수]
    def execute_model_lifecycle(type_name, save_name, iq_idx, style_idx, style_names):
        print(f"\n=============================================")
        print(f" [시작] {type_name} 모델 구축 및 학습")
        print(f"=============================================")
        
        # 1. 대상 파라미터를 Reg/Cls로 동적 분류
        reg_idx_local, cls_idx_local, num_cls_list = categorize_parameters_by_step(REF_FILE, style_names, STEP_THRESHOLD)
        
        # 2. y 데이터 슬라이싱
        y_tr_sub = y_train_scaled[:, style_idx]
        y_val_sub = y_val_scaled[:, style_idx]
        X_tr_sub = X_train_scaled[:, iq_idx]
        X_val_sub = X_val_scaled[:, iq_idx]
        
        # 3. 모델 초기화 및 학습
        model = SiameseStyleMultiTask_Base(
            input_dim=len(iq_idx), reg_dim=len(reg_idx_local), cls_num_classes_list=num_cls_list
        ).to(device)
        
        model = run_multitask_training_pipeline(
            model, X_tr_sub, y_tr_sub, X_val_sub, y_val_sub, 
            reg_idx_local, cls_idx_local, num_cls_list,
            batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE, device=device
        )
        
        torch.save(model.state_dict(), f'{save_name}.pth')
        
        # 4. 즉시 모델 평가
        evaluate_model(
            model=model, scaler_X=scaler_X, test_file_path=TEST_FILE, ref_file_path=REF_FILE,
            iq_indices=iq_idx, style_indices=style_idx, style_names=style_names,
            reg_indices_local=reg_idx_local, cls_indices_local=cls_idx_local, cls_num_classes_list=num_cls_list,
            output_file_path=f"eval_{save_name}.csv", restore=0, device=device, iq_dim=IQ_DIM
        )

    # ---------------------------------------------
    # [Type 1] 전체 통합
    execute_model_lifecycle("Type 1 Full", "model_Type1_Full", all_iq_indices, all_style_indices, style_cols)

    # [Type 3] Style 개별 모델
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        style_names = groups.get(lv, [])
        if not style_names: continue
        style_indices = [list(style_cols).index(c) for c in style_names]
        execute_model_lifecycle(f"Type 3 Style {lv}", f"model_Type3_Style_{lv}", all_iq_indices, style_indices, style_names)

    # [Type 4] IQ 개별 모델
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        iq_indices = iq_groups.get(lv, [])
        if not iq_indices: continue
        execute_model_lifecycle(f"Type 4 IQ {lv}", f"model_Type4_IQ_{lv}", iq_indices, all_style_indices, style_cols)

    # [Type 5] 1:1 매칭 모델
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        iq_indices = iq_groups.get(lv, [])
        style_names = groups.get(lv, [])
        if not iq_indices or not style_names: continue
        style_indices = [list(style_cols).index(c) for c in style_names]
        execute_model_lifecycle(f"Type 5 Matched {lv}", f"model_Type5_Matched_{lv}", iq_indices, style_indices, style_names)

    print("\n🎉 모든 파이프라인의 학습 및 평가가 성공적으로 종료되었습니다!")

if __name__ == "__main__":
    main()