import pandas as pd
import re

# ==========================================
# 1. 파일 경로 설정 (현재 환경에 맞게 파일명을 수정해 주세요)
# ==========================================
CAL_FILE = "image_quality_metrics_with_new_params_pyramidal_val.xlsx"  # 이미 계산이 완료된 엑셀 파일
PARAM_EXCEL_FILE = "Internal_Params_RZ20_Style1-7488.xlsx"             # Style 파라미터가 들어있는 정답지 엑셀 파일
OUTPUT_FILE = "final_merged_dataset.xlsx"                              # 새롭게 저장될 최종 엑셀 파일

def merge_dataframes():
    print("데이터를 불러오는 중입니다. 잠시만 기다려주세요...")
    try:
        # 엑셀 파일 로드 (데이터가 크면 수십 초 정도 소요될 수 있습니다)
        df_cal = pd.read_excel(CAL_FILE)
        df_param = pd.read_excel(PARAM_EXCEL_FILE)
    except Exception as e:
        print(f"파일 로드 중 오류가 발생했습니다: {e}")
        return

    # ==========================================
    # 2. PARAM_EXCEL_FILE의 'style' 컬럼을 인덱스로 설정
    # ==========================================
    # 대소문자나 공백 이슈를 방지하기 위해 'style' 컬럼을 유연하게 찾습니다.
    style_col_name = [col for col in df_param.columns if str(col).strip().lower() == 'style'][0]
    
    # style 컬럼을 정수형으로 변환 후, 검색(Merge)을 위해 인덱스로 세팅
    df_param[style_col_name] = df_param[style_col_name].astype(int)
    df_param = df_param.set_index(style_col_name)

    # ==========================================
    # 3. CAL_FILE의 첫 번째 컬럼에서 번호 추출 (Parsing)
    # ==========================================
    first_col_name = df_cal.columns[0] # 자동으로 첫 번째 컬럼(예: 'File Path')을 타겟으로 잡음
    
    def extract_style_number(filepath):
        # 'style'과 '.raw' 사이에 있는 숫자만 정확히 추출하는 정규표현식
        # (예: style1.raw, style_1.raw, style 1.raw 모두 호환)
        match = re.search(r'style[_\-\s]*(\d+)\.raw', str(filepath), re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    print(f"'{first_col_name}' 컬럼에서 Style 번호를 추출합니다...")
    # 추출한 숫자를 임시 컬럼('Match_Key')에 저장
    df_cal['Match_Key'] = df_cal[first_col_name].apply(extract_style_number)

    # 파싱 실패한 데이터 확인 및 경고
    unmatched_count = df_cal['Match_Key'].isna().sum()
    if unmatched_count > 0:
        print(f"⚠️ 경고: 파일명에서 Style 번호를 찾지 못한 행이 {unmatched_count}개 있습니다.")

    # ==========================================
    # 4. 데이터 병합 (Left Join)
    # ==========================================
    print("Style 파라미터를 계산된 데이터 뒷부분에 병합합니다...")
    # df_cal의 'Match_Key'와 df_param의 '인덱스(style 번호)'를 기준으로 병합
    df_merged = pd.merge(df_cal, df_param, left_on='Match_Key', right_index=True, how='left')

    # 병합이 끝났으므로 임시로 만든 Key 컬럼은 깔끔하게 삭제
    df_merged = df_merged.drop(columns=['Match_Key'])

    # ==========================================
    # 5. 최종 결과 저장
    # ==========================================
    print("최종 데이터를 엑셀로 저장하는 중입니다...")
    df_merged.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    
    print(f"✅ 작업 완료! 모든 데이터가 성공적으로 결합되어 '{OUTPUT_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    merge_dataframes()