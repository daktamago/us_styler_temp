import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# =====================================================================
# 1. 초기 엑셀 데이터 전처리 유틸리티 (필요시 호출)
# =====================================================================
def filter_excel_columns(input_file, output_file, columns_to_keep):
    df = pd.read_excel(input_file)
    actual_columns = [col for col in columns_to_keep if col in df.columns]
    filtered_df = df[actual_columns]
    filtered_df.to_excel(output_file, index=False)
    print(f"[전처리] 지정된 컬럼만 남기고 저장 완료: {output_file}")

def normalize_MINMAX(data_path, ref_path, target_cols, save_path):
    df = pd.read_excel(data_path, engine='openpyxl')
    ref_df = pd.read_excel(ref_path, engine='openpyxl', index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.capitalize()

    for col in target_cols:
        if col in df.columns and col in ref_df.columns:
            min_val = float(ref_df.loc['Min', col])
            max_val = float(ref_df.loc['Max', col])
            if max_val - min_val == 0:
                df[col] = 0.0
            else:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    df.to_excel(save_path, index=False, engine='openpyxl')
    print(f"[정규화] MinMax 정규화 완료: {save_path}")

def split_excel(file_path, train_save_path, test_save_path, test_size=0.1):
    df = pd.read_excel(file_path, engine='openpyxl')
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_excel(train_save_path, index=False, engine='openpyxl')
    test_df.to_excel(test_save_path, index=False, engine='openpyxl')
    print(f"[데이터 분할] Train/Test 분할 완료: Train({len(train_df)}), Test({len(test_df)})")


# =====================================================================
# 2. 모델 학습용 데이터 로드, 스케일링 및 그룹핑 (핵심 함수)
# =====================================================================
def load_scale_and_group_data(file_path, test_portion=0.2, random_state=42, iq_dim=60):
    """
    데이터를 로드하고, 학습/검증 셋으로 스플릿한 뒤 스케일링 및 Lv별 그룹핑을 수행합니다.
    """
    print(f"[{file_path}] 학습용 데이터 로드 및 전처리 시작...")
    
    ext = os.path.splitext(file_path)[1].lower()
    df = pd.read_excel(file_path) if ext in ['.xls', '.xlsx'] else pd.read_csv(file_path)
    
    # 1. IQ와 Style 컬럼 분리
    iq_cols = df.columns[:iq_dim]
    style_cols = df.columns[iq_dim:]
    
    X_raw = df.iloc[:, :iq_dim].values
    y_raw = df.iloc[:, iq_dim:].values
    
    # 2. Train / Val 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=test_portion, random_state=random_state
    )
    
    # 3. 스케일링 (X는 StandardScaler, y는 이미 정규화되어 있으므로 패스)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    y_train_scaled = y_train
    y_val_scaled = y_val
    scaler_y = None  # 구조 유지를 위해 반환
    
    # 4. 파라미터 그룹핑 (Lv0 ~ Lv3)
    iq_groups = {'Lv0': [], 'Lv1': [], 'Lv2': [], 'Lv3': []}
    for idx, col in enumerate(iq_cols):
        col_upper = str(col).strip().upper()
        if col_upper.endswith('LV1'): iq_groups['Lv1'].append(idx)
        elif col_upper.endswith('LV2'): iq_groups['Lv2'].append(idx)
        elif col_upper.endswith('LV3'): iq_groups['Lv3'].append(idx)
        else: iq_groups['Lv0'].append(idx) # 표시가 없으면 Lv0으로 간주
            
    style_groups = {'Lv1': [], 'Lv2': [], 'Lv3': [], 'Lv0': [], 'Others': []}
    for col in style_cols:
        col_upper = str(col).strip().upper()
        if col_upper.endswith('LV1'): style_groups['Lv1'].append(col)
        elif col_upper.endswith('LV2'): style_groups['Lv2'].append(col)
        elif col_upper.endswith('LV3'): style_groups['Lv3'].append(col)
        elif col_upper.endswith('LV0'): style_groups['Lv0'].append(col)
        else: style_groups['Lv0'].append(col) # 표시가 없으면 Lv0으로 간주
            
    print("✅ 데이터 로드, 스플릿, 스케일링 및 그룹핑 완료!\n")
    
    return (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, 
            scaler_X, scaler_y, 
            iq_groups, style_groups, style_cols, 
            len(iq_cols), len(style_cols))


# =====================================================================
# 칼럼 정보만 읽는 함수 데이터 로드, 스케일링 및 그룹핑 (핵심 함수)
# =====================================================================
def get_column_metadata(file_path, iq_dim=60):
    """
    평가(Evaluation) 시 사용하기 위해 데이터는 제외하고 컬럼명만 빠르게 읽어 그룹핑합니다.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # nrows=0 을 주면 데이터 행은 무시하고 첫 줄(Header)만 즉시 읽어옵니다. (연산 시간 0.1초)
    if ext in ['.xls', '.xlsx']:
        df_dummy = pd.read_excel(file_path, nrows=0) 
    else:
        df_dummy = pd.read_csv(file_path, nrows=0)
        
    iq_cols = df_dummy.columns[:iq_dim]
    style_cols = df_dummy.columns[iq_dim:]
    
    # 파라미터 그룹핑 (Lv0 ~ Lv3)
    iq_groups = {'Lv0': [], 'Lv1': [], 'Lv2': [], 'Lv3': []}
    for idx, col in enumerate(iq_cols):
        col_upper = str(col).strip().upper()
        if col_upper.endswith('LV1'): iq_groups['Lv1'].append(idx)
        elif col_upper.endswith('LV2'): iq_groups['Lv2'].append(idx)
        elif col_upper.endswith('LV3'): iq_groups['Lv3'].append(idx)
        else: iq_groups['Lv0'].append(idx)
            
    style_groups = {'Lv1': [], 'Lv2': [], 'Lv3': [], 'Lv0': [], 'Others': []}
    for col in style_cols:
        col_upper = str(col).strip().upper()
        if col_upper.endswith('LV1'): style_groups['Lv1'].append(col)
        elif col_upper.endswith('LV2'): style_groups['Lv2'].append(col)
        elif col_upper.endswith('LV3'): style_groups['Lv3'].append(col)
        elif col_upper.endswith('LV0'): style_groups['Lv0'].append(col)
        else: style_groups['Lv0'].append(col)
            
    return iq_groups, style_groups, style_cols, len(iq_cols), len(style_cols)