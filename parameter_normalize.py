import pandas as pd

def normalize_with_horizontal_ref(data_path, ref_path, target_cols, save_path):
    print(f"[{data_path}] 파일 정규화 작업 시작...")
    
    # 1. 파일 로드
    try:
        df = pd.read_excel(data_path, engine='openpyxl')
        
        # [수정된 부분] index_col=0 옵션을 주어, 불러올 때부터 첫 번째 열을 행 이름(인덱스)으로 고정합니다.
        ref_df = pd.read_excel(ref_path, engine='openpyxl', index_col=0)
    except Exception as e:
        print(f"파일 로드 중 오류가 발생했습니다: {e}")
        return

    # 2. 인덱스(행 이름) 강제 전처리
    # 문자열로 변환 -> 앞뒤 공백 제거 -> 첫 글자만 대문자로 변경 (예: ' MIN ', 'min' -> 모두 'Min'으로 통일)
    ref_df.index = ref_df.index.astype(str).str.strip().str.capitalize()

    # 'Min'과 'Max' 행이 존재하는지 체크 및 디버깅 메시지 추가
    if 'Min' not in ref_df.index or 'Max' not in ref_df.index:
        print("⚠️ 에러: 여전히 'Min' 또는 'Max' 행을 찾을 수 없습니다.")
        print(f"👉 힌트: 파이썬이 실제로 읽어들인 행 이름들은 다음과 같습니다: {ref_df.index.tolist()}")
        print("엑셀 파일 첫 열의 데이터가 위 리스트와 다르게 생겼다면 엑셀 파일 확인이 필요합니다.")
        return

    # 3. 지정된 패러미터에 대해 정규화 수행
    normalized_count = 0
    for col in target_cols:
        if col in df.columns and col in ref_df.columns:
            # Min, Max 값 추출
            min_val = float(ref_df.loc['Min', col])
            max_val = float(ref_df.loc['Max', col])
            
            # Max와 Min이 같은 경우 0으로 나누는 에러 방지
            if max_val - min_val == 0:
                print(f" - [경고] {col}: Min과 Max 값이 동일하여 0.0으로 일괄 변환합니다.")
                df[col] = 0.0
            else:
                # 정규화: (X - Min) / (Max - Min)
                df[col] = (df[col] - min_val) / (max_val - min_val)
                
            normalized_count += 1
        else:
            print(f" - [건너뜀] {col}: 데이터 파일이나 기준 파일에 해당 패러미터(Column)가 없습니다.")

    # 4. 정규화된 데이터 저장
    df.to_excel(save_path, index=False, engine='openpyxl')
    
    print(f"\n✅ 정규화 완료! (총 {normalized_count}개 컬럼 적용)")
    print(f" - 저장 위치: {save_path}")

# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 1. 학습/검증용 원본 데이터 경로
    TRAIN_DATA_FILE = "train_data_90.xlsx" 
    TEST_DATA_FILE = "test_data_10.xlsx"
    
    # 2. Min/Max 기준값이 들어있는 참조 파일 경로 (방금 알려주신 양식)
    MIN_MAX_REF_FILE = "example_minmax_ref.xlsx" 
    
    # 3. 정규화를 적용할 대상 패러미터 리스트 (정확한 Column 명을 입력하세요)
    TARGET_PARAMETERS = ["Edge Threshold-Lv2", "LapSmoothRate-Lv2"] 
    
    # 4. 정규화 완료 후 저장될 파일 이름
    TRAIN_OUTPUT_FILE = "train_data_90_normalized.xlsx"
    TEST_OUTPUT_FILE = "test_data_10_normalized.xlsx"

    # 학습용 데이터 정규화 실행
    # normalize_with_horizontal_ref(TRAIN_DATA_FILE, MIN_MAX_REF_FILE, TARGET_PARAMETERS, TRAIN_OUTPUT_FILE)
    
    # 검증용 데이터 정규화 실행 (학습 데이터와 동일한 Min/Max 기준 사용)
    # normalize_with_horizontal_ref(TEST_DATA_FILE, MIN_MAX_REF_FILE, TARGET_PARAMETERS, TEST_OUTPUT_FILE)