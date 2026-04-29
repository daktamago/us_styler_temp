import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_raw_data(raw_file, ref_file, target_columns, train_file, test_file, test_size=0.1, iq_dim=60):
    """
    [데이터 모드 1] 원본 데이터를 읽어 컬럼 필터링, 정규화, Train/Test 분할을 수행합니다.
    """
    print(f"🧹 [전처리] 원본 데이터 '{raw_file}' 정제 작업을 시작합니다...")
    df = pd.read_csv(raw_file)
    
    # 1. IQ 컬럼과 Target Style 컬럼 분리 및 유지
    iq_cols = list(df.columns[:iq_dim])
    actual_targets = [col for col in target_columns if col in df.columns]
    df = df[iq_cols + actual_targets]
    
    # 2. Min-Max 정규화
    ref_df = pd.read_csv(ref_file, index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.capitalize()
    
    for col in actual_targets:
        if col in ref_df.columns:
            min_val = float(ref_df.loc['Min', col])
            max_val = float(ref_df.loc['Max', col])
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
                
    # 3. Train / Test 분할 및 저장
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"✅ 전처리 완료! Train({len(train_df)}건) / Test({len(test_df)}건) 분할 저장 완료.\n")

def categorize_parameters_by_step(ref_file_path, current_style_cols, step_threshold=10):
    """
    Min/Max Reference 파일을 읽어, 전달받은 파라미터 리스트를 Regression과 Classification으로 분리합니다.
    """
    ref_df = pd.read_csv(ref_file_path, index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.capitalize()
    
    reg_indices, cls_indices, cls_num_classes = [], [], []
    
    for local_idx, col_name in enumerate(current_style_cols):
        if col_name in ref_df.columns:
            min_val = float(ref_df.loc['Min', col_name])
            max_val = float(ref_df.loc['Max', col_name])
            
            try:
                step_val = float(ref_df.loc['Step', col_name])
            except KeyError:
                reg_indices.append(local_idx)
                continue
                
            if step_val <= 0:
                 reg_indices.append(local_idx)
                 continue

            num_steps = (max_val - min_val) / step_val
            
            if num_steps <= step_threshold:
                cls_indices.append(local_idx)
                total_classes = int(num_steps * 2) + 1 # 차이값의 경우의 수
                cls_num_classes.append(total_classes)
            else:
                reg_indices.append(local_idx)
        else:
            reg_indices.append(local_idx)
            
    return reg_indices, cls_indices, cls_num_classes

def load_scale_and_group_data(file_path, test_portion=0.2, random_state=42, iq_dim=60):
    """학습용 Train 데이터를 로드하여 Scale 및 Lv 그룹핑 수행"""
    df = pd.read_csv(file_path)
    
    iq_cols = df.columns[:iq_dim]
    style_cols = df.columns[iq_dim:]
    
    X_raw = df.iloc[:, :iq_dim].values
    y_raw = df.iloc[:, iq_dim:].values
    
    X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=test_portion, random_state=random_state)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # 파라미터 그룹핑 (Lv0 ~ Lv3)
    iq_groups = {'Lv0': [], 'Lv1': [], 'Lv2': [], 'Lv3': []}
    for idx, col in enumerate(iq_cols):
        col_upper = str(col).strip().upper()
        if col_upper.endswith('LV1'): iq_groups['Lv1'].append(idx)
        elif col_upper.endswith('LV2'): iq_groups['Lv2'].append(idx)
        elif col_upper.endswith('LV3'): iq_groups['Lv3'].append(idx)
        else: iq_groups['Lv0'].append(idx)
            
    style_groups = {'Lv1': [], 'Lv2': [], 'Lv3': [], 'Lv0': []}
    for col in style_cols:
        col_upper = str(col).strip().upper()
        if col_upper.endswith('LV1'): style_groups['Lv1'].append(col)
        elif col_upper.endswith('LV2'): style_groups['Lv2'].append(col)
        elif col_upper.endswith('LV3'): style_groups['Lv3'].append(col)
        else: style_groups['Lv0'].append(col)
            
    return (X_train_scaled, X_val_scaled, y_train, y_val, scaler_X, 
            iq_groups, style_groups, style_cols, len(iq_cols), len(style_cols))