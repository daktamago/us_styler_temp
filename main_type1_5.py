import torch
import joblib

# 앞서 분리한 3개의 모듈 모두 Import
from data_processing import load_scale_and_group_data
from models import SiameseStyleRegressor_Base, SiameseStyleRegressor_MultiHead
from trainer import run_training_pipeline

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # =======================================================
    # [수정된 부분] 0. 원본 데이터에서 Test 데이터 완전히 분리하여 저장
    # =======================================================
    MASTER_FILE = "IQ_Style_remain_normalized.xlsx"  # 전체 원본 데이터 (정규화 완료된)
    TRAIN_FILE = "IQ_Style_Train.xlsx"               # 모델 학습/검증용으로 쓸 파일
    TEST_FILE = "IQ_Style_Test.xlsx"                 # 최종 평가 전용으로 숨겨둘 파일
    
    # 매번 새로 자르는 것을 방지하기 위해 파일이 없을 때만 분할하도록 처리
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("▶ 원본 데이터에서 Test용 데이터를 랜덤하게 분리합니다...")
        # 원본 데이터의 10%(0.1)를 Test용으로 잘라내서 저장합니다.
        split_excel(file_path=MASTER_FILE, train_save_path=TRAIN_FILE, test_save_path=TEST_FILE, test_size=0.1)
    
    (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, 
     scaler_X, scaler_y, 
     iq_groups, groups, style_cols, 
     total_iq_dim, total_style_dim) = load_scale_and_group_data(
         file_path=TRAIN_FILE, 
         test_portion=0.1, 
         random_state=42, 
         iq_dim=60
     )
    
    # 스케일러 저장 (나중에 Evaluation 할 때 사용)
    joblib.dump(scaler_X, 'scaler_x.pkl')
    if scaler_y: joblib.dump(scaler_y, 'scaler_y.pkl')

    # 하이퍼파라미터 설정
    batches = 256
    epochs = 50
    learning_r = 1e-3

    # =======================================================
    # [Type 1] 전체 통합 모델 (Input 60 -> Output 37)
    # =======================================================
    print("▶ [Type 1] 전체 통합 모델 학습 실행")
    model_type1 = SiameseStyleRegressor_Base(input_dim=total_iq_dim, output_dim=total_style_dim).to(device)
    trained_model_1 = run_training_pipeline(
        model=model_type1, 
        X_train=X_train_scaled, y_train=y_train_scaled, X_val=X_val_scaled, y_val=y_val_scaled,
        batch_size=batches, epochs=epochs, lr=learning_r, device=device
    )
    torch.save(trained_model_1.state_dict(), 'model_Type1_Full.pth')

    # =======================================================
    # [Type 2] 멀티헤드 모델 (Input 60 -> Output Head별 분할)
    # =======================================================
    print("▶ [Type 2] 멀티헤드 모델 학습 실행")
    head_dims = [len(groups[k]) for k in groups if len(groups[k]) > 0]
    
    model_type2 = SiameseStyleRegressor_MultiHead(input_dim=total_iq_dim, head_dims=head_dims).to(device)
    trained_model_2 = run_training_pipeline(
        model=model_type2, 
        X_train=X_train_scaled, y_train=y_train_scaled, X_val=X_val_scaled, y_val=y_val_scaled,
        batch_size=batches, epochs=epochs, lr=learning_r, device=device
    )
    torch.save(trained_model_2.state_dict(), 'model_Type2_MultiHead.pth')

    # =======================================================
    # [Type 3] Style 개별 모델 (Input 60 -> Output 각각의 Style Lv)
    # =======================================================
    print("▶ [Type 3] Style 개별 모델 학습 실행")
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        style_names = groups.get(lv, [])
        if not style_names: continue
            
        style_indices = [list(style_cols).index(c) for c in style_names]
        
        y_train_lv = y_train_scaled[:, style_indices]
        y_val_lv = y_val_scaled[:, style_indices]
        
        model_type3 = SiameseStyleRegressor_Base(input_dim=total_iq_dim, output_dim=len(style_indices)).to(device)
        trained_model_3 = run_training_pipeline(
            model=model_type3, 
            X_train=X_train_scaled, y_train=y_train_lv, X_val=X_val_scaled, y_val=y_val_lv,
            batch_size=batches, epochs=epochs, lr=learning_r, device=device
        )
        torch.save(trained_model_3.state_dict(), f'model_Type3_Style_{lv}.pth')

    # =======================================================
    # [Type 4] IQ 개별 모델 (Input 각각의 IQ Lv -> Output 전체)
    # =======================================================
    print("▶ [Type 4] IQ 개별 모델 학습 실행")
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        iq_indices = iq_groups.get(lv, [])
        if not iq_indices: continue
            
        X_train_lv = X_train_scaled[:, iq_indices]
        X_val_lv = X_val_scaled[:, iq_indices]
        
        model_type4 = SiameseStyleRegressor_Base(input_dim=len(iq_indices), output_dim=total_style_dim).to(device)
        trained_model_4 = run_training_pipeline(
            model=model_type4, 
            X_train=X_train_lv, y_train=y_train_scaled, X_val=X_val_lv, y_val=y_val_scaled,
            batch_size=batches, epochs=epochs, lr=learning_r, device=device
        )
        torch.save(trained_model_4.state_dict(), f'model_Type4_IQ_{lv}.pth')

    # =======================================================
    # [Type 5] 1:1 매칭 모델 (Input IQ Lv -> Output 동일 Style Lv)
    # =======================================================
    print("▶ [Type 5] 1:1 매칭 모델 학습 실행")
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        iq_indices = iq_groups.get(lv, [])
        style_names = groups.get(lv, [])
        
        if not iq_indices or not style_names: continue
            
        style_indices = [list(style_cols).index(c) for c in style_names]
        
        X_train_lv = X_train_scaled[:, iq_indices]
        X_val_lv = X_val_scaled[:, iq_indices]
        
        y_train_lv = y_train_scaled[:, style_indices]
        y_val_lv = y_val_scaled[:, style_indices]
        
        model_type5 = SiameseStyleRegressor_Base(input_dim=len(iq_indices), output_dim=len(style_indices)).to(device)
        trained_model_5 = run_training_pipeline(
            model=model_type5, 
            X_train=X_train_lv, y_train=y_train_lv, X_val=X_val_lv, y_val=y_val_lv,
            batch_size=batches, epochs=epochs, lr=learning_r, device=device
        )
        torch.save(trained_model_5.state_dict(), f'model_Type5_Matched_{lv}.pth')


if __name__ == "__main__":
    main()