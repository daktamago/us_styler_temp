import torch
import joblib
import os
import pandas as pd

from data_processing import prepare_raw_data, load_scale_and_group_data, categorize_parameters_by_step
from model import DynamicSiameseMultiTask
from trainer import run_multitask_training_pipeline
from evaluator import evaluate_model

def get_input(prompt, default_val):
    user_val = input(f"{prompt} (기본값: {default_val}): ").strip()
    return user_val if user_val else default_val

def get_user_options():
    print("="*50)
    print(" Pipeline Execution Options Setup ")
    print("="*50)
    
    # 파일 경로 설정
    raw_data_file = get_input("1. RAW_DATA_FILE 경로", "IQ_Style_Data.csv")
    train_file = get_input("2. TRAIN_FILE 경로", "IQ_Style_Train.csv")
    test_file = get_input("3. TEST_FILE 경로", "IQ_Style_Test.csv")
    scaler_path = get_input("4. SCALER_PATH 경로", "scaler_x.pkl")
    ref_file = get_input("5. Min/Max Reference 파일 경로", "ParameterMinMaxStep.csv")
    
    # 하이퍼파라미터 설정
    batch_size = int(get_input("6. BATCH_SIZE", "256"))
    epochs = int(get_input("7. EPOCHS", "50"))
    learning_rate = float(get_input("8. LEARNING_RATE", "0.001"))
    step_threshold = int(get_input("9. STEP_THRESHOLD", "10"))
    iq_dim = int(get_input("10. IQ Parameter(Input) 개수", "60"))

    # 데이터 전처리 여부
    prep_choice = input("11. 데이터 전처리를 수행하시겠습니까? (y/n, 기본 n): ").strip().lower()
    do_prep = prep_choice == 'y'
    
    target_col_file = None
    if do_prep:
        target_col_file = get_input("   - 추출할 타겟 컬럼 목록 파일 경로(CSV)", "target_cols.csv")
        
    # 모델 및 레이어 설정
    print("12. 히든 레이어 차원을 콤마(,)로 구분하여 입력하세요.")
    layers_input = input("    (기본값: 128,256,256): ").strip()
    hidden_dims = [int(x.strip()) for x in layers_input.split(',')] if layers_input else [128, 256, 256]
        
    # 학습 유형
    print("13. 학습 유형을 선택하세요.")
    print("    [1] 전체 Input -> 전체 Output (Type 1)")
    print("    [2] 전체 Input -> 각 Lv별 Output (Type 3)")
    train_type = get_input("    선택", "1")

    # Loss 가중치 설정
    print("14. 특정 파라미터(키워드)에 대한 Loss 가중치를 설정하세요.")
    print("    (예: Lv2:2.5, Lv3:2.5 / 미설정 시 엔터)")
    weight_input = input("    입력: ").strip()
    
    custom_weights = {}
    if weight_input:
        try:
            pairs = weight_input.split(',')
            for pair in pairs:
                k, v = pair.split(':')
                custom_weights[k.strip().upper()] = float(v.strip())
        except ValueError:
            print("    [Warning] 가중치 입력 형식이 올바르지 않아 가중치 없이 진행합니다.")

    # Evaluation Restore 설정
    restore = int(get_input("15. RESTORE (0: 정규화값으로 평가, 1: 원본스케일 복원 평가)", "0"))
    
    return {
        'raw_data_file': raw_data_file,
        'train_file': train_file,
        'test_file': test_file,
        'scaler_path': scaler_path,
        'ref_file': ref_file,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'step_threshold': step_threshold,
        'iq_dim': iq_dim,
        'do_prep': do_prep,
        'target_col_file': target_col_file,
        'hidden_dims': hidden_dims,
        'train_type': train_type,
        'custom_weights': custom_weights,
        'restore': restore
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = get_user_options()
    
    # [1] 데이터 전처리
    if opt['do_prep']:
        target_columns = []
        if opt['target_col_file'] and os.path.exists(opt['target_col_file']):
            target_df = pd.read_csv(opt['target_col_file'])
            target_columns = target_df.iloc[:, 0].tolist()
        
        prepare_raw_data(opt['raw_data_file'], opt['ref_file'], target_columns, 
                         opt['train_file'], opt['test_file'], test_size=0.1, iq_dim=opt['iq_dim'])

    # [2] 데이터 로드
    (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_X, 
     iq_groups, groups, style_cols, total_iq_dim, total_style_dim) = load_scale_and_group_data(opt['train_file'], iq_dim=opt['iq_dim'])
    
    joblib.dump(scaler_X, opt['scaler_path'])
    
    all_iq_indices = list(range(total_iq_dim))
    all_style_indices = list(range(total_style_dim))

    # [공통 학습/평가 실행 함수]
    def execute_model_lifecycle(type_name, save_name, iq_idx, style_idx, style_names):
        print(f"\n" + "="*45)
        print(f" [Start] {type_name} Model Training ")
        print("="*45)
        
        # 1. 대상 파라미터를 Reg/Cls로 동적 분류
        reg_idx_local, cls_idx_local, num_cls_list = categorize_parameters_by_step(opt['ref_file'], style_names, opt['step_threshold'])
        
        # 2. y 데이터 슬라이싱
        y_tr_sub = y_train_scaled[:, style_idx]
        y_val_sub = y_val_scaled[:, style_idx]
        X_tr_sub = X_train_scaled[:, iq_idx]
        X_val_sub = X_val_scaled[:, iq_idx]
        
        # 3. 모델 초기화 및 학습
        model = DynamicSiameseMultiTask(
            input_dim=len(iq_idx), hidden_dims=opt['hidden_dims'], 
            reg_dim=len(reg_idx_local), cls_num_classes_list=num_cls_list
        ).to(device)
        
        model = run_multitask_training_pipeline(
            model, X_tr_sub, y_tr_sub, X_val_sub, y_val_sub, 
            reg_idx_local, cls_idx_local, num_cls_list, style_names,
            custom_weights=opt['custom_weights'],
            batch_size=opt['batch_size'], epochs=opt['epochs'], lr=opt['learning_rate'], device=device
        )
        
        torch.save(model.state_dict(), f'{save_name}.pth')
        
        # 4. 모델 평가 (restore 값 유동적으로 적용)
        evaluate_model(
            model=model, scaler_X=scaler_X, test_file_path=opt['test_file'], ref_file_path=opt['ref_file'],
            iq_indices=iq_idx, style_indices=style_idx, style_names=style_names,
            reg_indices_local=reg_idx_local, cls_indices_local=cls_idx_local, cls_num_classes_list=num_cls_list,
            output_file_path=f"eval_{save_name}.csv", restore=opt['restore'], device=device, iq_dim=opt['iq_dim']
        )

    # ---------------------------------------------
    # [Type 1] 전체 통합
    if opt['train_type'] == '1':
        execute_model_lifecycle("Full Integrated (Type 1)", "model_Full", all_iq_indices, all_style_indices, style_cols)
    
    # [Type 3] Style 개별 모델
    else:
        for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
            style_names = groups.get(lv, [])
            if not style_names: continue
            style_indices = [list(style_cols).index(c) for c in style_names]
            execute_model_lifecycle(f"Level {lv} (Type 3)", f"model_Style_{lv}", all_iq_indices, style_indices, style_names)

    print("\n--- Pipeline Process Completed ---")

if __name__ == "__main__":
    main()