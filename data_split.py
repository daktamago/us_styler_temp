import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_and_split_excel(file_path, columns_to_drop, train_save_path, test_save_path):
    """
    엑셀 파일을 로드하여 특정 컬럼을 삭제하고, 데이터를 9:1로 무작위 분할하여 저장하는 함수
    
    Args:
        file_path (str): 원본 엑셀 파일 경로
        columns_to_drop (list): 삭제할 컬럼명 리스트
        train_save_path (str): 학습용 데이터(90%) 저장 경로
        test_save_path (str): 검증/테스트용 데이터(10%) 저장 경로
    """
    print(f"[{file_path}] 파일을 불러오는 중...")
    
    # 1. 엑셀 파일 로드 (엔진 지정)
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"데이터 로드 완료: 총 {len(df)} 행, {len(df.columns)} 열")
    except Exception as e:
        print(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
        return

    # 2. 지정된 Column 삭제
    # errors='ignore' 옵션을 주면 리스트에 없는 컬럼명이 섞여 있어도 에러를 뱉지 않고 넘깁니다.
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"컬럼 삭제 완료: 적용 후 {len(df.columns)} 열 남음")

    # 3. 데이터 무작위 9:1 분할 (Row 기준)
    # test_size=0.1은 10%를 테스트(검증)용으로 떼어낸다는 의미입니다.
    # random_state=42를 지정하여 나중에 코드를 다시 돌려도 똑같은 방식으로 분할되게 고정합니다.
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"데이터 분할 완료: 학습용 {len(train_df)} 행 / 검증용 {len(test_df)} 행")

    # 4. 분할된 데이터를 각각 새로운 엑셀 파일로 저장
    # index=False 로 설정해야 엑셀 파일에 불필요한 인덱스 번호 열이 추가되지 않습니다.
    train_df.to_excel(train_save_path, index=False, engine='openpyxl')
    test_df.to_excel(test_save_path, index=False, engine='openpyxl')
    
    print(f"\n저장 완료!")
    print(f"- 학습용 데이터: {train_save_path}")
    print(f"- 검증용 데이터: {test_save_path}")

# ==========================================
# 실제 실행 부분 (경로 및 컬럼명 수정 후 사용)
# ==========================================
if __name__ == "__main__":
    # 원본 파일 이름
    RAW_DATA_FILE = "original_data.xlsx" 
    
    # 삭제하고 싶은 컬럼명들을 리스트 형태로 입력하세요 (예: 불필요한 ID, 날짜 등)
    COLS_TO_DELETE = ["삭제할컬럼명1", "삭제할컬럼명2"] 
    
    # 저장될 파일 이름
    TRAIN_OUTPUT_FILE = "train_data_90.xlsx"
    TEST_OUTPUT_FILE = "test_data_10.xlsx"

    # 함수 실행
    # (주의: 코드 실행 전에 같은 폴더에 original_data.xlsx 파일이 있어야 합니다.)
    # preprocess_and_split_excel(RAW_DATA_FILE, COLS_TO_DELETE, TRAIN_OUTPUT_FILE, TEST_OUTPUT_FILE)