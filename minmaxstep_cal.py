import pandas as pd
import os

def analyze_csv_parameters(input_csv_path, output_csv_path):
    """
    CSV 파일을 읽어 각 수치형 컬럼의 Min, Max, StepSize를 계산하고 저장합니다.
    """
    # 1. 파일 존재 여부 확인
    if not os.path.exists(input_csv_path):
        print(f"오류: '{input_csv_path}' 파일을 찾을 수 없습니다.")
        return

    # 2. CSV 파일 로드
    print(f"'{input_csv_path}' 데이터를 불러오는 중...")
    df = pd.read_csv(input_csv_path)

    # 3. 숫자형(Numeric) 데이터 컬럼만 선택 (문자열 컬럼은 계산에서 제외)
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        print("오류: 계산할 수 있는 숫자형 데이터가 없습니다.")
        return

    # 4. Min, Max 계산
    min_vals = numeric_df.min()
    max_vals = numeric_df.max()

    # 5. StepSize 계산 (조건: (Max - Min) / 5)
    step_sizes = (max_vals - min_vals) / 5

    # 6. 결과를 담을 새로운 데이터프레임 생성
    result_df = pd.DataFrame({
        'Min': min_vals,
        'Max': max_vals,
        'StepSize': step_sizes
    })

    # 인덱스 이름 설정 (출력 양식의 첫 번째 빈 칸을 맞추기 위함)
    result_df.index.name = '' 

    # 7. 결과를 CSV로 저장
    result_df.to_csv(output_csv_path, float_format='%.6f') # 소수점 6자리까지 깔끔하게 출력
    
    print("="*50)
    print("분석 결과 미리보기:")
    print(result_df.head()) # 결과 일부 출력
    print("="*50)
    print(f"✅ 분석 완료! 결과가 '{output_csv_path}'에 성공적으로 저장되었습니다.")

# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 분석할 원본 CSV 파일 이름
    input_file = "input_data.csv"   # 실제 사용하는 파일명으로 수정하세요.
    
    # 결과를 저장할 CSV 파일 이름
    output_file = "parameter_info.csv"
    
    analyze_csv_parameters(input_file, output_file)