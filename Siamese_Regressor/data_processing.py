import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_raw_data(raw_file, ref_file, target_columns, train_file, test_file, test_size=0.1, iq_dim=80):
    print(f"\n[전처리] 원본 데이터 '{raw_file}' 정제 시작...")
    df = pd.read_csv(raw_file)
    iq_cols = list(df.columns[:iq_dim])
    all_style_cols = list(df.columns[iq_dim:])
    
    target_cols_cleaned = [str(c).strip().upper() for c in target_columns]
    actual_targets = [col for col in all_style_cols if str(col).strip().upper() in target_cols_cleaned]
    
    if len(actual_targets) == 0:
        raise ValueError("[Fatal Error] 추출할 Style 파라미터를 하나도 찾지 못했습니다!")

    filtered_df = df[iq_cols + actual_targets].copy()
    ref_df = pd.read_csv(ref_file, index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.upper()
    ref_col_map = {str(c).strip().upper(): c for c in ref_df.columns}

    for col in actual_targets:
        col_clean = str(col).strip().upper()
        if col_clean in ref_col_map:
            r_col = ref_col_map[col_clean]
            try:
                min_val, max_val = float(ref_df.loc['MIN', r_col]), float(ref_df.loc['MAX', r_col])
                filtered_df[col] = (filtered_df[col] - min_val) / (max_val - min_val) if max_val > min_val else 0.0
            except: pass

    train_df, test_df = train_test_split(filtered_df, test_size=test_size, random_state=42)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"  -> 전처리 완료! Train({len(train_df)}), Test({len(test_df)}) 분할 저장됨.")

def load_scale_and_group_data(file_path, iq_dim=80):
    df = pd.read_csv(file_path)
    iq_cols, style_cols = df.columns[:iq_dim], df.columns[iq_dim:]
    X_train, X_val, y_train, y_val = train_test_split(df[iq_cols].values, df[style_cols].values, test_size=0.1, random_state=42)
    scaler_X = StandardScaler()
    return scaler_X.fit_transform(X_train), scaler_X.transform(X_val), y_train, y_val, scaler_X, style_cols

def get_class_numbers(ref_file, style_names):
    ref_df = pd.read_csv(ref_file, index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.upper()
    ref_col_map = {str(c).strip().upper(): c for c in ref_df.columns}
    num_classes_list = []
    for col in style_names:
        c_clean = str(col).strip().upper()
        if c_clean in ref_col_map:
            r_col = ref_col_map[c_clean]
            try:
                min_val, max_val, step_val = float(ref_df.loc['MIN', r_col]), float(ref_df.loc['MAX', r_col]), float(ref_df.loc['STEP', r_col])
                num_classes_list.append(int(round((max_val - min_val) / step_val)) + 1 if step_val > 0 else 2)
            except: num_classes_list.append(2)
        else: num_classes_list.append(2)
    return num_classes_list\n