import cv2

import os

import re

import numpy as np

import pandas as pd

from scipy import ndimage

from scipy.signal import convolve2d

from concurrent.futures import ProcessPoolExecutor, as_completed





### Image Quality Factor Calculation Functions ###

def calculate_snr_with_speckle(image_16bit):

    if image_16bit is None or image_16bit.size == 0: return 0.0

    X = image_16bit.astype(np.float32)

    medianFiltered_image = ndimage.median_filter(X, size=5)

    deltaG_image = X - medianFiltered_image

    mean_medianFiltered_image = np.mean(medianFiltered_image)

    std_conventional = np.sqrt(np.mean(deltaG_image**2))

    if std_conventional < 1e-9: return float('inf') if mean_medianFiltered_image > 0 else 0.0

    snr_linear = mean_medianFiltered_image / std_conventional

    if snr_linear <= 0: return -float('inf')

    snr_db = 20 * np.log10(snr_linear)

    return snr_db



def calculate_contrast_rms(image_16bit):

    if image_16bit is None or image_16bit.size == 0: return 0.0

    return np.std(image_16bit, dtype=np.float64)



def calculate_weber_contrast(image_16bit):

    if image_16bit is None or image_16bit.size == 0: return 0.0

    image_float = image_16bit.astype(np.float32)

    background = ndimage.uniform_filter(image_float, size=7)

    background[background < 1.0] = 1.0

    weber_contrast = np.abs(image_float - background) / background

    return np.mean(weber_contrast)



def calculate_sharpness_laplacian(image_16bit):

    if image_16bit is None or image_16bit.size == 0: return 0.0

    laplacian = ndimage.laplace(image_16bit.astype(np.float32))

    return np.var(laplacian, dtype=np.float64)



def calculate_sharpness_brenner(image_16bit):

    if image_16bit is None or image_16bit.shape[1] < 3: return 0.0

    image_float = image_16bit.astype(np.float32)

    gradient_diff = image_float[:, 2:] - image_float[:, :-2]

    return np.mean(gradient_diff ** 2)



def calculate_speckle_index(image_16bit):

    if image_16bit is None or image_16bit.size == 0: return 0.0

    image_float = image_16bit.astype(np.float64)

    mean_val = np.mean(image_float)

    if mean_val < 1e-9: return 0.0

    std_val = np.std(image_float)

    return std_val / mean_val



def calculate_homogeneity(image_16bit):

    if image_16bit is None or image_16bit.size == 0: return 1.0

    min_val, max_val = np.min(image_16bit), np.max(image_16bit)

    if max_val <= min_val: return 1.0

    image_norm = ((image_16bit - min_val) * 31.0 / (max_val - min_val)).astype(np.uint8)

    glcm = np.zeros((32, 32), dtype=np.int32)

    pairs_i, pairs_j = image_norm[:, :-1].flatten(), image_norm[:, 1:].flatten()

    np.add.at(glcm, (pairs_i, pairs_j), 1)

    glcm_sum = glcm.sum()

    if glcm_sum == 0: return 1.0

    glcm_norm = glcm / glcm_sum

    i, j = np.ogrid[:32, :32]

    weights = 1.0 / (1.0 + np.abs(i - j))

    return np.sum(glcm_norm * weights)



def calculate_median_intensity(image_16bit):

    if image_16bit is None or image_16bit.size == 0: return 0.0

    return float(np.median(image_16bit))



def calculate_mode_intensity(image_16bit):

    if image_16bit is None or image_16bit.size == 0: return 0.0, 0.0

    hist, bin_edges = np.histogram(image_16bit, bins=256, range=(0, 65536))

    mode_bin_idx = np.argmax(hist)

    mode_intensity = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2.0

    second_mode_intensity = 0.0

    if len(hist) > 1:

        temp_hist = hist.copy()

        temp_hist[mode_bin_idx] = -1

        second_mode_bin_idx = np.argmax(temp_hist)

        second_mode_intensity = (bin_edges[second_mode_bin_idx] + bin_edges[second_mode_bin_idx + 1]) / 2.0

    return float(mode_intensity), float(second_mode_intensity)



def calculate_percentiles(image_16bit, percentile_25=25, percentile_75=75):

    if image_16bit is None or image_16bit.size == 0: return 0.0, 0.0

    p25, p75 = np.percentile(image_16bit, [percentile_25, percentile_75])

    return float(p25), float(p75)



def convert_value_to_8bit_fixed(value):

    

    # Convert 16 bit value to 8 bit value

    min_val = 0

    max_val = 2047  # 16 bit max value

    

    # Clipping the value b/w 0~2047

    clipped_value = np.clip(value, min_val, max_val)

    value_8bit = (clipped_value - min_val) * 255.0 / (max_val - min_val)

    

    # Clipping the final 8 bit value b/w 0~255

    return float(np.clip(value_8bit, 0.0, 255.0))



#  To use OpenCV functions image_16bit highly recommended to convert 8bit

def convert_16bit_to_8bit(image_16bit):



    # 16비트 값의 범위를 8비트(0-255)로 정규화

    if image_16bit.max() == image_16bit.min():

        return np.zeros_like(image_16bit, dtype=np.uint8)

    image_8bit = ((image_16bit - image_16bit.min()) / (image_16bit.max() - image_16bit.min()) * 255).astype(np.uint8)

    return image_8bit



def calculate_connectivity_auto_threshold(image_16bit):

    """

    오츠의 이진화 방법을 사용하여 자동으로 임계값을 결정하고 연결성 지표를 계산합니다.

    """

    if image_16bit is None or image_16bit.size == 0:

        return 0, 0.0, 0.0



    # 1. OpenCV에서 사용하기 위해 8비트로 변환

    image_8bit = convert_16bit_to_8bit(image_16bit)



    # 2. 오츠의 방법을 사용하여 자동으로 임계값 결정 및 이진화

    # threshold_value는 자동으로 결정된 임계값, binary_image는 결과 흑백 이미지

    threshold_value, binary_image = cv2.threshold(

        image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU

    )



    # 3. 연결 요소 분석 실행 (이후 과정은 이전과 동일)

    labeled_array, num_objects = ndimage.label(binary_image, structure=np.ones((3,3)))

    

    if num_objects == 0:

        return 0, 0.0, 0.0

        

    object_sizes = ndimage.sum_labels(binary_image, labeled_array, range(1, num_objects + 1))

    

    if object_sizes.size == 0:

        return 0, 0.0, 0.0



    total_object_pixels = np.sum(object_sizes)

    avg_object_size = np.mean(object_sizes)

    largest_object_size = np.max(object_sizes)

    largest_object_ratio = largest_object_size / total_object_pixels



    return float(num_objects), float(avg_object_size), float(largest_object_ratio)



def gaussian_downsample_scipy(image_16bit):

    # 1. 3x3 가우시안 커널 정의

    kernel = np.array([

        [1, 2, 1],

        [2, 4, 2],

        [1, 2, 1]

    ], dtype=np.float32) / 16.0

    

    # 2. 이미지에 커널을 적용하여 블러링 (Low-pass Filter 역할)

    # mode='same': 원본 이미지 크기 유지, boundary='symm': 엣지 부분 거울 반사 처리

    blurred_image = convolve2d(image_16bit, kernel, mode='same', boundary='symm')

    

    # 3. 2칸씩 건너뛰며 샘플링 (가로, 세로 해상도 1/2로 Downsampling)

    downsampled_image = blurred_image[::2, ::2]

    

    # 16bit 이미지 형식을 유지하기 위한 캐스팅

    return downsampled_image.astype(np.uint16)



def process_single_file(file_path, width, height, df_params):

    """

    단일 raw 파일에 대한 품질 지표(원본, LV1, LV2, LV3)와 파라미터 값을 계산하여 딕셔너리로 반환합니다.

    새롭게 추가된 연결성(Connectivity) 지표 3개도 함께 추출합니다.

    """

    

    #특정 이미지 배열에 대한 지표를 계산하고 딕셔너리 반환

    def extract_iq_metrics(img_array, suffix=""):

        # 1. 기존 품질 지표 계산 

        snr = calculate_snr_with_speckle(img_array)

        rms_contrast = calculate_contrast_rms(img_array)

        weber_contrast = calculate_weber_contrast(img_array)

        sharpness_lap = calculate_sharpness_laplacian(img_array)

        sharpness_bren = calculate_sharpness_brenner(img_array)

        speckle_idx = calculate_speckle_index(img_array)

        homogeneity = calculate_homogeneity(img_array)

        

        # 2. 밝기 지표 계산

        median_intensity = calculate_median_intensity(img_array)

        mode_intensity, mode2_intensity = calculate_mode_intensity(img_array)

        p25, p75 = calculate_percentiles(img_array)

        

        median_8bit = convert_value_to_8bit_fixed(median_intensity)

        mode_8bit = convert_value_to_8bit_fixed(mode_intensity)

        mode2_8bit = convert_value_to_8bit_fixed(mode2_intensity)

        p25_8bit = convert_value_to_8bit_fixed(p25)

        p75_8bit = convert_value_to_8bit_fixed(p75)

        

        # 3. 추가된 연결성(Connectivity) 지표 계산

        num_objects, avg_object_size, largest_object_ratio = calculate_connectivity_auto_threshold(img_array)

        

        # Suffix(_LV1, _LV2 등)를 붙여서 딕셔너리로 반환 (총 15개 항목)

        return {

            f"SNR{suffix}": snr,

            f"RMS Contrast{suffix}": rms_contrast,

            f"Contrast-Weber{suffix}": weber_contrast,

            f"Sharpness(Laplacian){suffix}": sharpness_lap,

            f"Sharpness(Brenner){suffix}": sharpness_bren,

            f"Speckle Index{suffix}": speckle_idx,

            f"Homogeneity{suffix}": homogeneity,

            f"Brightness-Median_8bit_fixed{suffix}": median_8bit,

            f"Brightness-Mode_8bit_fixed{suffix}": mode_8bit,

            f"Brightness-2nd Mode_8bit_fixed{suffix}": mode2_8bit,

            f"Brightness-Percentile_25th_8bit_fixed{suffix}": p25_8bit,

            f"Brightness-Percentile_75th_8bit_fixed{suffix}": p75_8bit,

            

            # 추가된 3개의 연결성 지표

            f"Connectivity-Num_Objects{suffix}": num_objects,

            f"Connectivity-Avg_Size{suffix}": avg_object_size,

            f"Connectivity-Largest_Ratio{suffix}": largest_object_ratio,

        }



    try:

        with open(file_path, 'rb') as f:

            raw_data = np.fromfile(f, dtype='<u2')

        

        if raw_data.size != width * height:

            print(f"  - 경고: '{os.path.basename(file_path)}' 파일 크기({raw_data.size})가 폴더명의 크기 정보({width*height})와 일치하지 않습니다. 건너뜁니다.")

            return None

            

        # 1. 원본 이미지 (LV0)

        image_16bit = raw_data.reshape((height, width))

        

        # 2. 다운샘플링 이미지 생성 (LV1, LV2, LV3)

        image_lv1 = gaussian_downsample_scipy(image_16bit)

        image_lv2 = gaussian_downsample_scipy(image_lv1)

        image_lv3 = gaussian_downsample_scipy(image_lv2)

        

        # 3. 각 해상도별 지표 추출 및 통합

        quality_metrics = {"File Path": file_path}

        

        # 딕셔너리 업데이트(.update)를 통해 지표들을 하나로 합침

        # (원본 15개 + LV1 15개 + LV2 15개 + LV3 15개 = 파일 경로 제외 총 60개 지표)

        quality_metrics.update(extract_iq_metrics(image_16bit, suffix=""))     # 원본

        quality_metrics.update(extract_iq_metrics(image_lv1, suffix="_LV1"))   # 1번 다운샘플링

        quality_metrics.update(extract_iq_metrics(image_lv2, suffix="_LV2"))   # 2번 다운샘플링

        quality_metrics.update(extract_iq_metrics(image_lv3, suffix="_LV3"))   # 3번 다운샘플링

        

        # 4. 파일명에서 Style 번호 매칭 및 파라미터 가져오기

        filename = os.path.basename(file_path)

        style_match = re.search(r'style_(\d+)', filename, re.IGNORECASE)

        

        param_values = {}

        if style_match:

            style_number = int(style_match.group(1))

            try:

                param_series = df_params.loc[style_number]

                param_values = param_series.to_dict()

            except KeyError:

                print(f"  - 경고: '{filename}'의 style({style_number})에 해당하는 파라미터를 찾을 수 없습니다.")

        else:

            print(f"  - 경고: '{filename}' 파일명에서 style 번호를 찾을 수 없습니다.")

        

        # 최종 결과 병합 및 반환

        return {**quality_metrics, **param_values}

        

    except Exception as e:

        print(f"  - 오류: '{os.path.basename(file_path)}' 처리 중 문제 발생 - {e}")

        return None



def process_folders(root_path, param_file_path, output_file):

    try:

        print(f"파라미터 파일 로딩 시작: {param_file_path}")

        df_params = pd.read_excel(param_file_path).set_index('style')

        print("파라미터 파일 로딩 완료.")

    except Exception as e:

        print(f"오류: 파라미터 파일을 읽는 중 문제가 발생했습니다 - {e}")

        return



    results, tasks = [], []

    

    print(f"지정된 경로 탐색 시작: {root_path}")

    if not os.path.isdir(root_path):

        print(f"오류: 지정된 경로 '{root_path}'를 찾을 수 없습니다.")

        return



    for dirpath, _, filenames in os.walk(root_path):

    

        #Must not contain the training data

        #Origin



        if dirpath == r'C:\workspace_medical\ValidationData\260223_StyleUS_forTest\test_data\3VV PA\SDMR_Input_w1008_h394_14_36_14_338' :

            continue



        match = re.search(r'_w(\d+)_h(\d+)_', os.path.basename(dirpath))

        if not match: continue

        width, height = int(match.group(1)), int(match.group(2))

        

        for filename in filenames:

            if filename.lower().endswith('.raw') and 'style' in filename.lower():

                tasks.append((os.path.join(dirpath, filename), width, height, df_params))

    

    if not tasks:

        print("\n처리할 유효한 '.raw' 파일을 찾지 못했습니다.")

        return

    print(f"총 {len(tasks)}개의 파일을 처리합니다. (병렬 처리 시작)")

    

    with ProcessPoolExecutor() as executor:

        future_to_file = {executor.submit(process_single_file, *task): task[0] for task in tasks}

        for i, future in enumerate(as_completed(future_to_file), 1):

            if result := future.result(): results.append(result)

            print(f"  - 진행률: {i}/{len(tasks)} 처리 완료.", end='\r')



    print("\n모든 파일 처리 완료. 결과를 저장합니다.")

    

    if not results:

        print("처리된 결과가 없어 파일을 저장하지 않습니다.")

        return

        

    try:

        # 데이터프레임 생성 및 정렬

        df_results = pd.DataFrame(results).sort_values(by="File Path").reset_index(drop=True)

        

        # 1. 15개의 베이스 품질 지표 이름 정의 (원본)

        base_metrics = [

            "SNR", 

            "RMS Contrast", 

            "Contrast-Weber", 

            "Sharpness(Laplacian)", 

            "Sharpness(Brenner)", 

            "Speckle Index", 

            "Homogeneity", 

            "Brightness-Median_8bit_fixed", 

            "Brightness-Mode_8bit_fixed", 

            "Brightness-2nd Mode_8bit_fixed", 

            "Brightness-Percentile_25th_8bit_fixed", 

            "Brightness-Percentile_75th_8bit_fixed",

            # Connectivity 지표 3개

            "Connectivity-Num_Objects",

            "Connectivity-Avg_Size",

            "Connectivity-Largest_Ratio"

        ]

        

        # 2. LV0(원본)부터 LV3까지의 Suffix 리스트

        suffixes = ["", "_LV1", "_LV2", "_LV3"]

        

        # 3. File Path를 맨 앞에 두고, 베이스 지표 * 4개 레벨(총 60개) 컬럼 리스트 자동 생성

        quality_cols = ["File Path"]

        for suffix in suffixes:

            for metric in base_metrics:

                quality_cols.append(f"{metric}{suffix}")

        

        # 4. Style 파라미터 컬럼 추가

        param_cols = df_params.columns.tolist()

        

        # DataFrame에 존재하는 컬럼만 최종 순서 리스트에 포함 (안전 장치)

        final_cols_order = [col for col in quality_cols if col in df_results.columns] + \

                           [col for col in param_cols if col in df_results.columns]

        

        # 5. 엑셀 파일로 저장

        df_results[final_cols_order].to_excel(output_file, index=False, engine='openpyxl')

        

        print(f"\n작업 완료! 모든 결과가 '{output_file}' 파일에 저장되었습니다.")

        

    except Exception as e:

        print(f"\n오류: 결과를 파일에 저장하는 중 문제가 발생했습니다 - {e}")



if __name__ == '__main__':

    #Need to Change PATHs

    ROOT_FOLDER_PATH = r"C:\workspace_medical\ValidationData\260223_StyleUS_forTest\test_data"

    PARAMS_EXCEL_FILE = "Internal_Params_RZ20_Style1-7488.xlsx"

    # ▼▼▼ --- 수정된 부분 (출력 파일명 변경) --- ▼▼▼

    OUTPUT_EXCEL_FILE = "image_quality_metrics_with_params_pyramidal_val.xlsx"

    # ▲▲▲ --- 수정된 부분 --- ▲▲▲

    

    process_folders(ROOT_FOLDER_PATH, PARAMS_EXCEL_FILE, OUTPUT_EXCEL_FILE)

