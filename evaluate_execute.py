import torch
import joblib
import os
import numpy as np
from models import SiameseStyleRegressor_Base, SiameseStyleRegressor_MultiHead
from data_processing import load_scale_and_group_data
from evaluator import evaluate_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 통합 평가 시스템 가동 (Device: {device})")

    # 1. 경로 및 설정 세팅
    TEST_FILE = "IQ_Style_Test.xlsx"
    REF_FILE = "example_minmax_ref.xlsx"
    TRAIN_FILE = "IQ_Style_Train.xlsx"  # 그룹핑 정보 획득용
    SCALER_PATH = 'scaler_x.pkl'
    
    if not os.path.exists(TEST_FILE):
        print(f"❌ 에러: 테스트 파일({TEST_FILE})이 없습니다.")
        return

    # 2. 메타데이터 및 그룹핑 정보 로드
    # (평가 시에는 데이터 스플릿이 필요 없으므로 아주 작은 부분만 로드해서 구조만 파악)
    _, _, _, _, scaler_X, _, iq_groups, style_groups, style_cols, total_iq_dim, total_style_dim = \
        load_scale_and_group_data(TRAIN_FILE, test_portion=0.01, iq_dim=60)
    
    # 저장된 실제 스케일러 로드
    if os.path.exists(SCALER_PATH):
        scaler_X = joblib.load(SCALER_PATH)
    else:
        print(f"⚠️ 경고: {SCALER_PATH}가 없어 새로 생성된 스케일러를 사용합니다. 결과가 부정확할 수 있습니다.")

    all_iq_indices = list(range(total_iq_dim))
    all_style_indices = list(range(total_style_dim))

    # =================================================================
    # [Type 1] 전체 통합 모델 평가
    # =================================================================
    model_path = 'model_Type1_Full.pth'
    if os.path.exists(model_path):
        print("\n[평가 진행] Type 1: Full Model")
        model = SiameseStyleRegressor_Base(input_dim=total_iq_dim, output_dim=total_style_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        evaluate_model(model, scaler_X, TEST_FILE, REF_FILE, all_iq_indices, all_style_indices, 
                       style_cols, "eval_Type1_Full.xlsx", device=device)

    # =================================================================
    # [Type 2] Multi-Head 모델 평가
    # =================================================================
    model_path = 'model_Type2_MultiHead.pth'
    if os.path.exists(model_path):
        print("\n[평가 진행] Type 2: Multi-Head Model")
        # Head 순서에 맞게 스타일 인덱스 재배열 (Lv1 -> Lv2 -> Lv3 -> Lv0 -> Others 순서)
        head_ordered_indices = []
        head_ordered_names = []
        head_dims = []
        
        for lv in ['Lv1', 'Lv2', 'Lv3', 'Lv0', 'Others']:
            names = style_groups.get(lv, [])
            if names:
                indices = [list(style_cols).index(c) for c in names]
                head_ordered_indices.extend(indices)
                head_ordered_names.extend(names)
                head_dims.append(len(names))
        
        model = SiameseStyleRegressor_MultiHead(input_dim=total_iq_dim, head_dims=head_dims).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        evaluate_model(model, scaler_X, TEST_FILE, REF_FILE, all_iq_indices, head_ordered_indices, 
                       head_ordered_names, "eval_Type2_MultiHead.xlsx", device=device)

    # =================================================================
    # [Type 3] Style 개별 모델 평가 (Lv0~Lv3 반복)
    # =================================================================
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        model_path = f'model_Type3_Style_{lv}.pth'
        if os.path.exists(model_path):
            print(f"\n[평가 진행] Type 3: Style {lv} 전용 모델")
            target_names = style_groups[lv]
            target_indices = [list(style_cols).index(c) for c in target_names]
            
            model = SiameseStyleRegressor_Base(input_dim=total_iq_dim, output_dim=len(target_indices)).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            evaluate_model(model, scaler_X, TEST_FILE, REF_FILE, all_iq_indices, target_indices, 
                           target_names, f"eval_Type3_Style_{lv}.xlsx", device=device)

    # =================================================================
    # [Type 4] IQ 개별 모델 평가 (Lv0~Lv3 반복)
    # =================================================================
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        model_path = f'model_Type4_IQ_{lv}.pth'
        if os.path.exists(model_path):
            print(f"\n[평가 진행] Type 4: IQ {lv} 입력 전용 모델")
            target_iq_indices = iq_groups[lv]
            
            model = SiameseStyleRegressor_Base(input_dim=len(target_iq_indices), output_dim=total_style_dim).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            evaluate_model(model, scaler_X, TEST_FILE, REF_FILE, target_iq_indices, all_style_indices, 
                           style_cols, f"eval_Type4_IQ_{lv}.xlsx", device=device)

    # =================================================================
    # [Type 5] 1:1 매칭 모델 평가 (Lv0~Lv3 반복)
    # =================================================================
    for lv in ['Lv0', 'Lv1', 'Lv2', 'Lv3']:
        model_path = f'model_Type5_Matched_{lv}.pth'
        if os.path.exists(model_path):
            print(f"\n[평가 진행] Type 5: Matched {lv} 모델")
            target_iq_indices = iq_groups[lv]
            target_style_names = style_groups[lv]
            target_style_indices = [list(style_cols).index(c) for c in target_style_names]
            
            model = SiameseStyleRegressor_Base(input_dim=len(target_iq_indices), output_dim=len(target_style_indices)).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            evaluate_model(model, scaler_X, TEST_FILE, REF_FILE, target_iq_indices, target_style_indices, 
                           target_style_names, f"eval_Type5_Matched_{lv}.xlsx", device=device)

    print("\n✨ 모든 모델의 평가가 완료되었습니다. 생성된 엑셀 파일들을 확인하세요.")

if __name__ == "__main__":
    main()