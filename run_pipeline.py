import torch
import joblib
import os
import pandas as pd

from data_processing import prepare_raw_data, load_scale_and_group_data, categorize_parameters_by_step
from model import DynamicSiameseMultiTask
from trainer import run_multitask_training_pipeline
from evaluator import evaluate_model

def get_user_options():
    print("="*50)
    print(" 🛠️  파이프라인 실행 옵션 설정")
    print("="*50)
    
    # (1) 데이터 전처리 여부
    prep_choice = input("1. 데이터 전처리를 수행하시겠습니까? (y/n): ").strip().lower()
    do_prep = prep_choice == 'y'
    
    ref_file = "ParameterMinMaxStep.csv"
    target_col_file = None
    if do_prep:
        ref_file = input("   - Min/Max Reference 파일 경로 (예: ParameterMinMaxStep.csv): ").strip()
        target_col_file = input("   - 추출할 타겟 컬럼 목록 파일 경로 (CSV, 예: target_cols.csv): ").strip()
        
    # (2) IQ Parameter 개수
    iq_dim_input = input("2. Input(IQ parameter)의 개수를 입력하세요 (예: 60): ").strip()
    iq_dim = int(iq_dim_input) if iq_dim_input.isdigit() else 60
    
    # (3) 학습 모델 선택
    print("3. 학습 모델을 선택하세요.")
    print("   [1] Siamese Network (현재)")
    model_choice = input("   선택: ").strip()
    
    # (4) Layer 숫자 및 차원
    print("4. 히든 레이어 차원을 콤마(,)로 구분하여 입력하세요.")
    layers_input = input("   (기본값 사용 시 엔터, 예: 128,256,256): ").strip()
    hidden_dims = [int(x.strip()) for x in layers_input.split(',')] if layers_input else [128, 256, 256]
        
    # (5) 학습 유형
    print("5. 학습 유형을 선택하세요.")
    print("   [1] 전체 Input -> 전체 Output (Type 1)")
    print("   [2] 전체 Input -> 각 Lv별 Output (Type 3)")
    train_type = input("   선택: ").strip()

    # (6) 🔥 동적 Loss 가중치 설정 (신규 추가)
    print("6. 특정 파라미터(키워드)에 대한 Loss 가중치를 설정하시겠습니까?")
    print("   (입력 예시: Lv2:2.5, Lv3:2.5, Edge:1.5 / 미설정 시 엔터)")
    weight_input = input("   입력: ").strip()
    
    custom_weights = {}
    if weight_input:
        try:
            pairs = weight_input.split(',')
            for pair in pairs:
                k, v = pair.split(':')
                # 대소문자 구분을 없애기 위해 키를 대문자로 통일
                custom_weights[k.strip().upper()] = float(v.strip())
        except ValueError:
            print("   ⚠️ 가중치 입력 형식이 올바르지 않아 기본값(가중치 없음)으로 진행합니다.")
            custom_weights = {}
    
    return {
        'do_prep': do_prep,
        'ref_file': ref_file,
        'target_col_file': target_col_file,
        'iq_dim': iq_dim,
        'model_choice': model_choice,
        'hidden_dims': hidden_dims,
        'train_type': train_type,
        'custom_weights': custom_weights # 딕셔너리 형태로 저장
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options = get_user_options()
    
    RAW_DATA_FILE = "IQ_Style_Data.csv"
    TRAIN_FILE = "IQ_Style_Train.csv"
    TEST_FILE = "IQ_Style_Test.csv"
    SCALER_PATH = 'scaler_x.pkl'
    
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    STEP_THRESHOLD = 10
    
    # [1] 데이터 전처리
    if options['do_prep']:
        # CSV 파일에서 타겟 컬럼명 리스트 추출
        target_columns = []
        if options['target_col_file'] and os.path.exists(options['target_col_file']):
            target_df = pd.read_csv(options['target_col_file'])
            # 첫 번째 컬럼에 타겟 이름들이 있다고 가정
            target_columns = target_df.iloc[:, 0].tolist() 
        else:
            print("⚠️ 타겟 컬럼 파일을 찾을 수 없어 기본 모드로 진행합니다.")
            
        prepare_raw_data(RAW_DATA_FILE, options['ref_file'], target_columns, TRAIN_FILE, TEST_FILE, test_size=0.1, iq_dim=options['iq_dim'])

# [2] 데이터 로드
    (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, scaler_X, 
     iq_groups, groups, style_cols, total_iq_dim, total_style_dim) = load_scale_and_group_data(TRAIN_FILE, iq_dim=options['iq_dim'])
    
    joblib.dump(scaler_X, SCALER_PATH)
    
    all_iq_indices = list(range(total_iq_dim))
    all_style_indices = list(range(total_style_dim))

    # [공통 실행 함수]
    def execute_model_lifecycle(type_name, save_name, iq_idx, style_idx, style_names, hidden_dims, custom_weights):
        print(f"\n=============================================")
        print(f" [시작] {type_name} 모델 학습 (레이어: {hidden_dims})")
        if custom_weights:
            print(f"   * 적용된 커스텀 가중치: {custom_weights}")
        print(f"=============================================")
        
        reg_idx_local, cls_idx_local, num_cls_list = categorize_parameters_by_step(options['ref_file'], style_names, STEP_THRESHOLD)
        
        y_tr_sub = y_train_scaled[:, style_idx]
        y_val_sub = y_val_scaled[:, style_idx]
        X_tr_sub = X_train_scaled[:, iq_idx]
        X_val_sub = X_val_scaled[:, iq_idx]
        
        model = DynamicSiameseMultiTask(
            input_dim=len(iq_idx), hidden_dims=hidden_dims, 
            reg_dim=len(reg_idx_local), cls_num_classes_list=num_cls_list
        ).to(device)
        
        # 🔥 custom_weights를 trainer로 전달
        model = run_multitask_training_pipeline(
            model, X_tr_sub, y_tr_sub, X_val_sub, y_val_sub, 
            reg_idx_local, cls_idx_local, num_cls_list, style_names,
            custom_weights=custom_weights, # 파라미터 추가
            batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE, device=device
        )
        
        torch.save(model.state_dict(), f'{save_name}.pth')
        
        evaluate_model(
            model=model, scaler_X=scaler_X, test_file_path=TEST_FILE, ref_file_path=options['ref_file'],
            iq_indices=iq_idx, style_indices=style_idx, style_names=style_names,
            reg_indices_local=reg_idx_local, cls_indices_local=cls_idx_local, cls_num_classes_list=num_cls_list,
            output_file_path=f"eval_{save_name}.csv", restore=0, device=device, iq_dim=options['iq_dim']
        )

    # [3] 학습 분기
    if options['train_type'] == '1':
        execute_model_lifecycle("전체 통합 (Full)", "model_Full_to_Full", all_iq_indices, all_style_indices, style_cols, options['hidden_dims'], options['custom_weights'])
    elif options['train_type'] == '2':
        for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
            style_names = groups.get(lv, [])
            if not style_names: continue
            style_indices = [list(style_cols).index(c) for c in style_names]
            execute_model_lifecycle(f"각 Lv별 (Style {lv})", f"model_Full_to_Style_{lv}", all_iq_indices, style_indices, style_names, options['hidden_dims'], options['custom_weights'])

    print("\n🎉 파이프라인 실행 완료!")

if __name__ == "__main__":
    main()