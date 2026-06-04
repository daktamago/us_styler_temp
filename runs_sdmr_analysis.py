import os
import shutil
import subprocess
import re
import numpy as np
import pandas as pd
from iq_calculator import extract_full_iq_ref  # 별도 작성한 파일에서 함수 가져오기

def main():
    # 폴더 및 파일 경로 설정 (코드 실행 폴더 기준)
    rsdmr_styles_dir = os.path.join("RSDMR_generator", "styles")
    
    style_gen_dir = "Style_generator"
    style_gen_styles_dir = os.path.join(style_gen_dir, "styles")
    style_gen_raw_dir = os.path.join(style_gen_dir, "RawData")
    style_gen_rawout_dir = os.path.join(style_gen_dir, "RawDataOut") # 새로 분석할 대상 폴더
    
    refer_raw_dir = "Refer_Raw"
    exe_name = "SDMR_RUN.exe"
    exe_path = os.path.join(style_gen_dir, exe_name)

    # 타겟 폴더들이 존재하는지 확인 (없으면 생성)
    os.makedirs(style_gen_styles_dir, exist_ok=True)
    os.makedirs(style_gen_raw_dir, exist_ok=True)
    os.makedirs(style_gen_rawout_dir, exist_ok=True)

    # RSDMR_generator\styles 폴더 내의 모든 파일 목록 가져오기
    if not os.path.exists(rsdmr_styles_dir):
        print(f">> [오류] 폴더를 찾을 수 없습니다: {rsdmr_styles_dir}")
        return

    style_files = [f for f in os.listdir(rsdmr_styles_dir) if os.path.isfile(os.path.join(rsdmr_styles_dir, f))]

    for style_file in style_files:
        print(f"\n==================================================")
        print(f">> 처리 시작: {style_file}")
        print(f"==================================================")

        # ---------------------------------------------------------
        # 1. 스타일 파일을 \Style_generator\styles 로 복사
        # ---------------------------------------------------------
        src_style_path = os.path.join(rsdmr_styles_dir, style_file)
        dst_style_path = os.path.join(style_gen_styles_dir, style_file)
        shutil.copy2(src_style_path, dst_style_path)
        print(f"[1] 스타일 파일 복사 완료: {style_file}")

        # ---------------------------------------------------------
        # 2. 파일명에서 확장자, 뒤의 '_style####', 맨 앞의 'style' 제거 후 복사
        # ---------------------------------------------------------
        base_name_only = os.path.splitext(style_file)[0]
        
        # 1) 뒤에 붙은 '_style숫자' 패턴 삭제
        raw_base_name = re.sub(r'_style\d+', '', base_name_only)
        
        # 2) 맨 앞에 있는 'style' 문자열 삭제 (style 뒤에 언더바(_)가 하나 있을 경우 같이 삭제)
        raw_base_name = re.sub(r'^style_?', '', raw_base_name)
        
        raw_file_name = f"{raw_base_name}.raw"
        src_raw_path = os.path.join(refer_raw_dir, raw_file_name)
        dst_raw_path = os.path.join(style_gen_raw_dir, raw_file_name)

        if os.path.exists(src_raw_path):
            shutil.copy2(src_raw_path, dst_raw_path)
            print(f"[2] Raw 파일 추출 및 복사 완료: {raw_file_name}")
        else:
            print(f"[오류] 일치하는 Raw 파일을 찾을 수 없습니다: {src_raw_path}")
            print(">> 이 파일의 처리를 건너뛰고 찌꺼기 파일을 정리합니다.")
            os.remove(dst_style_path)
            continue

        # ---------------------------------------------------------
        # 3. \Style_generator 폴더에서 SDMR_RUN.exe 실행
        # ---------------------------------------------------------
        if os.path.exists(exe_path):
            print(f"[3] {exe_name} 실행 중...")
            try:
                # WinError 2 방지를 위해 절대 경로로 변환
                abs_exe_path = os.path.abspath(exe_path)
                abs_cwd_path = os.path.abspath(style_gen_dir)
                
                # exe 파일이 Style_generator 폴더 내의 상대경로를 참조할 수 있도록 cwd(작업 디렉토리) 변경
                subprocess.run([abs_exe_path], cwd=abs_cwd_path, check=True)
                print(f"[3] {exe_name} 정상 종료")
            except subprocess.CalledProcessError as e:
                print(f"[오류] {exe_name} 실행 중 문제 발생: {e}")


        # ---------------------------------------------------------
        # 4. \Style_generator\styles 와 \Style_generator\RawData 파일 전부 삭제
        # ---------------------------------------------------------
        print("[4] 사용된 파일 정리 중...")
        for f in os.listdir(style_gen_styles_dir):
            os.remove(os.path.join(style_gen_styles_dir, f))
        for f in os.listdir(style_gen_raw_dir):
            os.remove(os.path.join(style_gen_raw_dir, f))
        
        print("[4] 폴더 내 파일 삭제 완료")

    print("\n>> 모든 파일에 대한 반복 작업이 완료되었습니다.")
    
    # ---------------------------------------------------------
    # 5. \Style_generator\RawDataOut 및 하위폴더 파일들의 IQ 파라미터 계산
    # ---------------------------------------------------------
    print(f"\n>> {style_gen_rawout_dir} 및 하위 폴더의 IQ 파라미터 분석을 시작합니다.")
    if os.path.exists(style_gen_rawout_dir):
        iq_results = []
        
        # os.walk를 사용하여 하위 폴더까지 전부 탐색
        for root, dirs, files in os.walk(style_gen_rawout_dir):
            for raw_f in files:
                if not raw_f.endswith('.raw'):
                    continue  # .raw 파일이 아니면 건너뜀
                
                raw_path = os.path.join(root, raw_f)
                
                # 파일명 정리: 'styleSDMR'을 찾고 그 앞의 문자는('style' 포함) 제거
                idx = raw_f.find('styleSDMR')
                if idx != -1:
                    # 'styleSDMR'에서 'style'(5글자)를 건너뛰고 'SDMR'부터 남김
                    clean_name = raw_f[idx + 5:] 
                else:
                    clean_name = raw_f
                    
                # 이미지 차원 파싱
                w_match = re.search(r'w(\d+)', raw_f)
                h_match = re.search(r'h(\d+)', raw_f)
                width = int(w_match.group(1)) if w_match else 720
                height = int(h_match.group(1)) if h_match else 249
                
                try:
                    # 16비트 raw 이미지 로드
                    raw_img = np.fromfile(raw_path, dtype=np.uint16).reshape((height, width))
                    
                    # IQ 계산 (불러온 모듈 활용)
                    iq_dict = extract_full_iq_ref(raw_img)
                    
                    # 딕셔너리의 제일 앞에 FileName이 오도록 새 딕셔너리 구성
                    iq_row = {'FileName': clean_name}
                    iq_row.update(iq_dict)
                    iq_results.append(iq_row)
                    
                    print(f"   - [분석 완료] {clean_name}")
                    
                except Exception as e:
                    print(f"   - [오류] {raw_path} 파일 분석 중 에러 발생: {e}")
                
        # 결과 CSV 저장
        if iq_results:
            df_iq = pd.DataFrame(iq_results)
            out_csv_path = "RawDataOut_IQ_Parameters.csv"
            df_iq.to_csv(out_csv_path, index=False)
            print(f"\n>> [결과 저장 완료] 총 {len(iq_results)}개 파일의 IQ 계산 결과가 {out_csv_path} 에 기록되었습니다.")
        else:
            print(f">> [알림] 분석할 .raw 파일을 찾지 못했습니다.")
            
        # ---------------------------------------------------------
        # 6. RawDataOut_IQ_Parameters.csv 와 input_raw_iq.csv 비교 (Difference 계산)
        # ---------------------------------------------------------
        raw_iq_path = "RawDataOut_IQ_Parameters.csv"
        input_iq_path = "input_raw_iq.csv"
        
        print(f"\n>> {raw_iq_path} 와 {input_iq_path} 의 파라미터 비교 분석을 시작합니다.")
        
        if os.path.exists(raw_iq_path) and os.path.exists(input_iq_path):
            try:
                df_raw_iq = pd.read_csv(raw_iq_path)
                df_input_iq = pd.read_csv(input_iq_path)
                
                # input_raw_iq.csv 의 첫 번째 행(Row) 데이터를 타겟 기준으로 사용
                input_iq_target = df_input_iq.iloc[0]
                
                # 두 CSV 파일 간의 공통 컬럼(Column Head) 찾기
                common_cols = [col for col in df_raw_iq.columns 
                               if col in df_input_iq.columns and pd.api.types.is_numeric_dtype(df_raw_iq[col])]
                
                df_diff = df_raw_iq.copy()
                
                for col in common_cols:
                    df_diff[col] = df_raw_iq[col] - input_iq_target[col]
                
                out_diff_path = "pred_iq_diff.csv"
                df_diff.to_csv(out_diff_path, index=False)
                print(f">> [결과 저장 완료] Difference 계산 결과가 {out_diff_path} 에 성공적으로 기록되었습니다.")
                
            except Exception as e:
                print(f">> [오류] Difference 계산 중 에러 발생: {e}")
        else:
            if not os.path.exists(raw_iq_path): print(f">> [알림] {raw_iq_path} 파일이 없습니다.")
            if not os.path.exists(input_iq_path): print(f">> [알림] {input_iq_path} 파일이 없습니다.")

        # ---------------------------------------------------------
        # 7. [신규 추가] Predicted_IQ_Difference.csv(input_iqs_diff) 와 pred_iq_diff.csv 결과값 비교
        # ---------------------------------------------------------
        # 사용자가 지칭한 input_iqs_diff.csv는 style_est_only2.py에서 생성되는 Predicted_IQ_Difference.csv 매칭
        input_diff_path = "Predicted_IQ_Difference.csv"
        if not os.path.exists(input_diff_path) and os.path.exists("input_iqs_diff.csv"):
            input_diff_path = "input_iqs_diff.csv"
            
        pred_diff_path = "pred_iq_diff.csv"
        if not os.path.exists(pred_diff_path) and os.path.exists("pred_iqs_diff.csv"):
            pred_diff_path = "pred_iqs_diff.csv"

        print(f"\n>> {input_diff_path} 와 {pred_diff_path} 의 결과값 비교 매칭을 시작합니다.")
        
        if os.path.exists(input_diff_path) and os.path.exists(pred_diff_path):
            try:
                df_input_diff = pd.read_csv(input_diff_path)
                df_pred_diff = pd.read_csv(pred_diff_path)
                
                # 1. StyleNum 무시 (비교 전 제거)
                if 'StyleNum' in df_input_diff.columns:
                    df_input_diff = df_input_diff.drop(columns=['StyleNum'])
                if 'StyleNum' in df_pred_diff.columns:
                    df_pred_diff = df_pred_diff.drop(columns=['StyleNum'])
                
                # 두 파일 간의 공통 숫자 파라미터 컬럼 찾기
                common_params = [col for col in df_input_diff.columns 
                                 if col in df_pred_diff.columns and col != 'FileName' and pd.api.types.is_numeric_dtype(df_input_diff[col])]
                
                # FileName을 기준으로 두 데이터 병합 (동일 파일명 매칭)
                df_merged = pd.merge(df_input_diff, df_pred_diff, on='FileName', suffixes=('_input', '_pred'))
                
                compare_results = []
                
                for idx, row in df_merged.iterrows():
                    # 각 파라미터별 4가지 항목을 하나의 FileName 아래에 4개의 행(Row)으로 나누어 기록 (example.csv 양식 적용)
                    row_input = {'FileName': row['FileName'], 'Type': 'Input_Diff'}
                    row_pred = {'FileName': '', 'Type': 'Pred_Diff'}
                    row_diff = {'FileName': '', 'Type': 'Input-Pred'}
                    row_abs = {'FileName': '', 'Type': 'abs(Input-Pred)'}
                    
                    for col in common_params:
                        val_in = row[f"{col}_input"]
                        val_pr = row[f"{col}_pred"]
                        diff_val = val_in - val_pr
                        abs_diff = abs(diff_val)
                        
                        row_input[col] = val_in
                        row_pred[col] = val_pr
                        row_diff[col] = diff_val
                        row_abs[col] = abs_diff
                        
                    compare_results.extend([row_input, row_pred, row_diff, row_abs])
                
                df_compare = pd.DataFrame(compare_results)
                
                # 'Type' 컬럼명을 비워두어 example.csv와 동일한 헤더 구조(FileName,,SNR,...) 성립
                df_compare = df_compare.rename(columns={'Type': ''})
                
                out_compare_path = "iq_diff_compare.csv"
                df_compare.to_csv(out_compare_path, index=False)
                
                print(f">> [결과 저장 완료] 두 파일의 매칭 및 비교 결과가 {out_compare_path} 에 성공적으로 기록되었습니다.")
                
            except Exception as e:
                print(f">> [오류] 최종 매칭 및 비교 중 에러 발생: {e}")
        else:
            print(">> [알림] 비교할 CSV 파일이 누락되어 매칭 분석을 건너뜁니다.")

if __name__ == "__main__":
    main()