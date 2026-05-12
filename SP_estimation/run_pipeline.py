import torch
import joblib
import os
import pandas as pd

from data_processing import prepare_raw_data, load_scale_and_group_data, categorize_parameters_by_step
from model import DynamicDirectMultiTask # 수정된 모델
from trainer import run_multitask_training_pipeline
from evaluator import evaluate_model

def get_input(prompt, default_val):
    user_val = input(f"{prompt} (기본값: {default_val}): ").strip()
    return user_val if user_val else default_val

def get_user_options():
    print("="*50)
    print(" Direct Prediction Setup ")
    print("="*50)
    raw_data_file = get_input("1. RAW_DATA_FILE", "IQ_Style_Data.csv")
    train_file = get_input("2. TRAIN_FILE", "IQ_Style_Train.csv")
    test_file = get_input("3. TEST_FILE", "IQ_Style_Test.csv")
    scaler_path = get_input("4. SCALER_PATH", "scaler_x.pkl")
    ref_file = get_input("5. Min/Max Reference", "ParameterMinMaxStep.csv")
    batch_size = int(get_input("6. BATCH_SIZE", "256"))
    epochs = int(get_input("7. EPOCHS", "50"))
    learning_rate = float(get_input("8. LEARNING_RATE", "0.001"))
    iq_dim = int(get_input("9. IQ Parameter 개수", "60"))
    
    print("10. Encoder 히든 레이어 (콤마 구분)")
    hidden_dims = [int(x.strip()) for x in get_input("    차원", "128,256,512").split(',')]
    print("11. Extractor 히든 레이어 (콤마 구분)")
    extractor_dims = [int(x.strip()) for x in get_input("    차원", "512,256").split(',')]
    print("12-R. Regressor Head 히든 레이어 (콤마 구분)")
    reg_head_dims = [int(x.strip()) for x in get_input("    차원", "512,128").split(',')]
    print("12-C. Classifier Head 히든 레이어 (콤마 구분)")
    cls_head_dims = [int(x.strip()) for x in get_input("    차원", "256").split(',')]
    
    custom_weights = {} # 가중치 로직 생략(필요시 이전과 동일하게 추가 가능)
    restore = int(get_input("13. RESTORE (0/1)", "0"))
    eval_filename = get_input("14. 결과 파일명", "direct_test_results")

    return locals() # 모든 지역 변수를 딕셔너리로 반환 (단순화)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = get_user_options()
    
    (X_train, X_val, y_train, y_val, scaler_X, _, groups, style_cols, total_iq, total_style) = load_scale_and_group_data(opt['train_file'], iq_dim=opt['iq_dim'])
    joblib.dump(scaler_X, opt['scaler_path'])
    
    iq_idx = list(range(total_iq))
    style_idx = list(range(total_style))

    print(f"\n--- [Start] Direct Prediction Model Training ---")
    reg_idx, cls_idx, num_cls = categorize_parameters_by_step(opt['ref_file'], style_cols, 10)
    
    model = DynamicDirectMultiTask(
        input_dim=total_iq, hidden_dims=opt['hidden_dims'], 
        extractor_dims=opt['extractor_dims'], reg_head_dims=opt['reg_head_dims'], 
        cls_head_dims=opt['cls_head_dims'], reg_dim=len(reg_idx), cls_num_classes_list=num_cls
    ).to(device)
    
    model = run_multitask_training_pipeline(
        model, X_train, y_train, X_val, y_val, reg_idx, cls_idx, num_cls, style_cols,
        custom_weights={}, batch_size=opt['batch_size'], epochs=opt['epochs'], lr=opt['learning_rate'], device=device
    )
    
    torch.save(model.state_dict(), 'direct_model.pth')
    evaluate_model(model, scaler_X, opt['test_file'], opt['ref_file'], iq_idx, style_idx, style_cols,
                   reg_idx, cls_idx, num_cls, f"{opt['eval_filename']}.csv", opt['restore'], device=device, iq_dim=opt['iq_dim'])

if __name__ == "__main__":
    main()