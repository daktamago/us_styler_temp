import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_raw_data(raw_file, ref_file, target_columns, train_file, test_file, test_size=0.1, iq_dim=60):
    print(f"\n[데이터 전처리] 원본 데이터 '{raw_file}' 정제 시작...")
    
    # 1. 원본 데이터 로드
    df = pd.read_csv(raw_file)
    
    # IQ 파라미터는 무조건 앞의 iq_dim 개수만큼 가져옴
    iq_cols = list(df.columns[:iq_dim])
    all_style_cols = list(df.columns[iq_dim:])
    
    print(f"  - 원본 컬럼 분석: 총 {len(df.columns)}개 (IQ: {len(iq_cols)}개, Style: {len(all_style_cols)}개)")

    # 2. 타겟 컬럼 필터링 (대소문자 무시, 양끝 공백 제거로 완벽 매칭)
    target_cols_cleaned = [str(c).strip().upper() for c in target_columns]
    actual_targets = []
    
    for col in all_style_cols:
        if str(col).strip().upper() in target_cols_cleaned:
            actual_targets.append(col)
            
    print(f"  - 타겟 파일과 매칭된 대상 Style 파라미터: {len(actual_targets)}개")
    
    # [안전장치] 매칭된 타겟이 0개일 경우 강제 종료 및 안내
    if len(actual_targets) == 0:
        raise ValueError(
            "\n[Fatal Error] 추출할 Style 파라미터를 하나도 찾지 못했습니다!\n"
            "원인: '추출할 컬럼 목록 파일'의 이름과 원본 데이터의 컬럼 이름이 일치하지 않거나 파일이 비어있습니다.\n"
        )

    filtered_df = df[iq_cols + actual_targets].copy()

    # 3. Min-Max 정규화 수행
    print(f"  - Min/Max Reference 파일('{ref_file}')을 기반으로 정규화 수행 중...")
    ref_df = pd.read_csv(ref_file, index_col=0)
    
    # Reference 파일의 인덱스(Min, Max 등)와 컬럼명을 맵핑하기 쉽게 정제 (.str.upper() 로 에러 수정)
    ref_df.index = ref_df.index.astype(str).str.strip().str.upper()
    ref_col_map = {str(c).strip().upper(): c for c in ref_df.columns}

    normalized_count = 0
    for col in actual_targets:
        col_clean = str(col).strip().upper()
        if col_clean in ref_col_map:
            ref_col_name = ref_col_map[col_clean]
            try:
                min_val = float(ref_df.loc['MIN', ref_col_name])
                max_val = float(ref_df.loc['MAX', ref_col_name])
                if max_val > min_val:
                    filtered_df[col] = (filtered_df[col] - min_val) / (max_val - min_val)
                    normalized_count += 1
                else:
                    filtered_df[col] = 0.0 # MAX와 MIN이 같은 경우
            except KeyError:
                pass # Min/Max 인덱스가 없으면 스킵
                
    print(f"  - 정규화 성공: {normalized_count}/{len(actual_targets)}개 파라미터")

    # 4. Train / Test 분할 및 저장
    train_df, test_df = train_test_split(filtered_df, test_size=test_size, random_state=42)
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"  - 전처리 완료! Train({len(train_df)}행), Test({len(test_df)}행) 분할 저장됨.\n")


def load_scale_and_group_data(file_path, iq_dim=60):
    df = pd.read_csv(file_path)
    
    iq_cols = df.columns[:iq_dim]
    style_cols = df.columns[iq_dim:]
    
    X_raw = df[iq_cols].values
    y_raw = df[style_cols].values
    
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.1, random_state=42)
    
    # IQ 파라미터 정규화 (Z-score)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # 파라미터 그룹핑
    iq_groups = {'Lv0': [], 'Lv1': [], 'Lv2': [], 'Lv3': []}
    for idx, col in enumerate(iq_cols):
        col_upper = str(col).strip().upper()
        if col_upper.endswith('LV1') or '_LV1' in col_upper: iq_groups['Lv1'].append(idx)
        elif col_upper.endswith('LV2') or '_LV2' in col_upper: iq_groups['Lv2'].append(idx)
        elif col_upper.endswith('LV3') or '_LV3' in col_upper: iq_groups['Lv3'].append(idx)
        else: iq_groups['Lv0'].append(idx)
            
    groups = {'Lv0': [], 'Lv1': [], 'Lv2': [], 'Lv3': []}
    for col in style_cols:
        col_upper = str(col).strip().upper()
        if col_upper.endswith('LV1') or '_LV1' in col_upper: groups['Lv1'].append(col)
        elif col_upper.endswith('LV2') or '_LV2' in col_upper: groups['Lv2'].append(col)
        elif col_upper.endswith('LV3') or '_LV3' in col_upper: groups['Lv3'].append(col)
        else: groups['Lv0'].append(col)
        
    return X_train_scaled, X_val_scaled, y_train, y_val, scaler_X, iq_groups, groups, style_cols, len(iq_cols), len(style_cols)


def categorize_parameters_by_step(ref_file, style_names, step_threshold=10):
    ref_df = pd.read_csv(ref_file, index_col=0)
    
    # Reference 파일 인덱스 정제 (.str.upper() 로 에러 수정)
    ref_df.index = ref_df.index.astype(str).str.strip().str.upper()
    ref_col_map = {str(c).strip().upper(): c for c in ref_df.columns}
    
    reg_indices = []
    cls_indices = []
    cls_num_classes_list = []
    
    for idx, col in enumerate(style_names):
        col_clean = str(col).strip().upper()
        is_reg = True 
        
        if col_clean in ref_col_map:
            ref_col_name = ref_col_map[col_clean]
            try:
                step_val = float(ref_df.loc['STEP', ref_col_name])
                min_val = float(ref_df.loc['MIN', ref_col_name])
                max_val = float(ref_df.loc['MAX', ref_col_name])
                
                if step_val > 0:
                    num_classes = int(round((max_val - min_val) / step_val)) + 1
                    if num_classes <= step_threshold:
                        is_reg = False
                        cls_indices.append(idx)
                        cls_num_classes_list.append(num_classes)
            except (KeyError, ValueError):
                pass 
                
        if is_reg:
            reg_indices.append(idx)
            
    return reg_indices, cls_indices, cls_num_classes_list