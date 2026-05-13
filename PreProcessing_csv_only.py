import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog

# UI 기반 파일 선택 헬퍼 함수
def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw() # 메인 윈도우 숨김
    root.attributes('-topmost', True) # 다이얼로그를 항상 최상단에 표시
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def prepare_raw_data(raw_file, ref_file, target_col_file, train_file, test_file, test_size=0.1, iq_dim=80):
    print(f"\n[Processing] 원본 데이터 '{os.path.basename(raw_file)}' 정제 시작...")
    
    # 1. 타겟 컬럼 읽기 (가로/세로 방향 모두 자동 대응)
    target_columns = []
    if os.path.exists(target_col_file):
        target_df = pd.read_csv(target_col_file, header=None)
        if target_df.shape[1] > 1 and target_df.shape[0] == 1:
            target_columns = target_df.iloc[0, :].astype(str).tolist()
        else:
            target_columns = target_df.iloc[:, 0].astype(str).tolist()
    else:
        raise FileNotFoundError(f"타겟 컬럼 파일을 찾을 수 없습니다: {target_col_file}")

    # 2. 원본 데이터 로드
    df = pd.read_csv(raw_file)
    
    iq_cols = list(df.columns[:iq_dim])
    all_style_cols = list(df.columns[iq_dim:])
    
    print(f"  - 원본 컬럼 분석: 총 {len(df.columns)}개 (IQ: {len(iq_cols)}개, Style: {len(all_style_cols)}개)")

    # 3. 타겟 컬럼 필터링 (공백 및 대소문자 방어 로직 적용)
    target_cols_cleaned = [str(c).strip().upper() for c in target_columns]
    actual_targets = []
    
    for col in all_style_cols:
        if str(col).strip().upper() in target_cols_cleaned:
            actual_targets.append(col)
            
    print(f"  - 타겟 파일과 매칭된 대상 Style 파라미터: {len(actual_targets)}개")
    
    if len(actual_targets) == 0:
        raise ValueError(
            "\n[Fatal Error] 추출할 Style 파라미터를 하나도 찾지 못했습니다!\n"
            "원인: '추출할 컬럼 목록 파일'의 이름과 원본 데이터의 컬럼 이름이 일치하지 않거나 파일이 비어있습니다."
        )

    filtered_df = df[iq_cols + actual_targets].copy()

    # 4. Min-Max 정규화 수행 (Reference 파일 기준)
    print(f"  - Min/Max Reference 파일 기반으로 정규화 수행 중...")
    ref_df = pd.read_csv(ref_file, index_col=0)
    
    # Reference 인덱스 정리 (.str.upper() 에러 픽스 적용)
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

    # 5. Train / Test 분할 및 저장
    train_df, test_df = train_test_split(filtered_df, test_size=test_size, random_state=42)
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"  - 전처리 완료! Train({len(train_df)}행), Test({len(test_df)}행) 분할 저장됨.\n")

def main():
    print("="*60)
    print(" Standalone Data Preprocessing Script (UI Version) ")
    print("="*60)
    print("파일 선택 창이 열립니다. (화면에 보이지 않으면 작업 표시줄을 확인하세요.)\n")

    csv_types = (("CSV files", "*.csv"), ("All files", "*.*"))

    # 1. 파일 선택 (UI 팝업)
    print("1. 원본 데이터(Raw Data) 파일을 선택하세요...")
    raw_data_file = select_file("원본 데이터(Raw Data) 파일 선택", csv_types)
    if not raw_data_file: return print("취소되었습니다.")
    print(f"  -> 선택됨: {os.path.basename(raw_data_file)}\n")

    print("2. 타겟 컬럼 목록 파일(Target Columns)을 선택하세요...")
    target_col_file = select_file("타겟 컬럼 목록 파일 선택", csv_types)
    if not target_col_file: return print("취소되었습니다.")
    print(f"  -> 선택됨: {os.path.basename(target_col_file)}\n")

    print("3. Min/Max Reference 파일을 선택하세요...")
    ref_file = select_file("Min/Max Reference 파일 선택", csv_types)
    if not ref_file: return print("취소되었습니다.")
    print(f"  -> 선택됨: {os.path.basename(ref_file)}\n")

    # 2. 기타 설정 입력 (터미널)
    iq_dim_str = input("4. IQ Parameter(Input) 개수를 입력하세요 (기본값: 80): ").strip()
    iq_dim = int(iq_dim_str) if iq_dim_str else 80

    test_size_str = input("5. Test 데이터 분할 비율을 입력하세요 (예: 0.1 -> 10%, 기본값: 0.1): ").strip()
    test_size = float(test_size_str) if test_size_str else 0.1

    train_file = input("6. 생성될 Train 파일 이름을 입력하세요 (기본값: Train_Processed.csv): ").strip()
    if not train_file: train_file = "Train_Processed.csv"

    test_file = input("7. 생성될 Test 파일 이름을 입력하세요 (기본값: Test_Processed.csv): ").strip()
    if not test_file: test_file = "Test_Processed.csv"

    # 3. 실행
    try:
        prepare_raw_data(raw_data_file, ref_file, target_col_file, train_file, test_file, test_size, iq_dim)
    except Exception as e:
        print(f"\n[오류 발생] {e}")

if __name__ == "__main__":
    main()