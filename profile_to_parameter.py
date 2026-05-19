import os
import sys
import re
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from scipy.signal import convolve2d
# import torch  # PyTorch 모델을 사용할 경우 주석 해제
# import joblib # Scaler를 로드할 경우 주석 해제

# =====================================================================
# 1. 기존 & 신규 Image Quality(IQ) 계산 함수 (new_param_cal.py 기반)
# =====================================================================
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
    return 20 * np.log10(snr_linear)

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
    return np.std(image_float) / mean_val

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
    min_val, max_val = 0, 2047
    clipped_value = np.clip(value, min_val, max_val)
    value_8bit = (clipped_value - min_val) * 255.0 / (max_val - min_val)
    return float(np.clip(value_8bit, 0.0, 255.0))

def convert_16bit_to_8bit(image_16bit):
    if image_16bit.max() == image_16bit.min():
        return np.zeros_like(image_16bit, dtype=np.uint8)
    return ((image_16bit - image_16bit.min()) / (image_16bit.max() - image_16bit.min()) * 255).astype(np.uint8)

def calculate_connectivity_auto_threshold(image_16bit):
    if image_16bit is None or image_16bit.size == 0: return 0, 0.0, 0.0
    image_8bit = convert_16bit_to_8bit(image_16bit)
    _, binary_image = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled_array, num_objects = ndimage.label(binary_image, structure=np.ones((3,3)))
    if num_objects == 0: return 0, 0.0, 0.0
    object_sizes = ndimage.sum_labels(binary_image, labeled_array, range(1, num_objects + 1))
    if object_sizes.size == 0: return 0, 0.0, 0.0
    return float(num_objects), float(np.mean(object_sizes)), float(np.max(object_sizes) / np.sum(object_sizes))

def calculate_directional_gradients(image_16bit):
    if image_16bit is None or image_16bit.size == 0: return 0.0, 0.0
    img_float = image_16bit.astype(np.float32)
    sobel_x = ndimage.sobel(img_float, axis=1)
    sobel_y = ndimage.sobel(img_float, axis=0)
    return float(np.mean(np.abs(sobel_x))), float(np.mean(np.abs(sobel_y)))

def calculate_gradient_extremes(image_16bit, percentile=90):
    if image_16bit is None or image_16bit.size == 0: return 0.0
    img_float = image_16bit.astype(np.float32)
    sobel_x = ndimage.sobel(img_float, axis=1)
    sobel_y = ndimage.sobel(img_float, axis=0)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.percentile(grad_mag, percentile))

def calculate_masked_cv(image_16bit, bottom_percentile=20):
    if image_16bit is None or image_16bit.size == 0: return 0.0
    img_float = image_16bit.astype(np.float32)
    sobel_x = ndimage.sobel(img_float, axis=1)
    sobel_y = ndimage.sobel(img_float, axis=0)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    thresh = np.percentile(grad_mag, bottom_percentile)
    mask = grad_mag <= thresh
    if not np.any(mask): return 0.0
    flat_pixels = img_float[mask]
    mean_val = np.mean(flat_pixels)
    if mean_val < 1e-9: return 0.0
    return float(np.std(flat_pixels) / mean_val)

def calculate_st_ten(image_16bit, blur_sigma=1.0, tau_percentile=50):
    if image_16bit is None or image_16bit.size == 0: return 0.0
    img_float = image_16bit.astype(np.float32)
    img_blur = ndimage.gaussian_filter(img_float, sigma=blur_sigma)
    sobel_x = ndimage.sobel(img_blur, axis=1)
    sobel_y = ndimage.sobel(img_blur, axis=0)
    grad_sq = sobel_x**2 + sobel_y**2
    grad_mag = np.sqrt(grad_sq)
    tau = np.percentile(grad_mag, tau_percentile)
    st_ten_val = np.sum(grad_sq[grad_mag >= tau]) / (img_float.size + 1e-9)
    return float(st_ten_val)

def extract_base_iq_metrics(img_array):
    """Raw 이미지에서 참조용 기본(Base) IQ(IQ_ref)를 계산합니다."""
    median_intensity = calculate_median_intensity(img_array)
    mode_intensity, mode2_intensity = calculate_mode_intensity(img_array)
    p25, p75 = calculate_percentiles(img_array)
    num_objects, avg_object_size, largest_object_ratio = calculate_connectivity_auto_threshold(img_array)
    grad_x, grad_y = calculate_directional_gradients(img_array)

    return {
        "SNR": calculate_snr_with_speckle(img_array),
        "Speckle Index": calculate_speckle_index(img_array),
        "Homogeneity": calculate_homogeneity(img_array),
        "Non-Edge_CV": calculate_masked_cv(img_array),
        "RMS Contrast": calculate_contrast_rms(img_array),
        "Contrast-Weber": calculate_weber_contrast(img_array),
        "ST-Ten": calculate_st_ten(img_array),
        "Sharpness(Laplacian)": calculate_sharpness_laplacian(img_array),
        "Sharpness(Brenner)": calculate_sharpness_brenner(img_array),
        "Connectivity-Num_Objects": num_objects,
        "Connectivity-Avg_Size": avg_object_size,
        "Connectivity-Largest_Ratio": largest_object_ratio,
        "Gradient_X": grad_x,
        "Gradient_Y": grad_y,
        "Brightness-Median_8bit_fixed": convert_value_to_8bit_fixed(median_intensity),
        "Brightness-Mode_8bit_fixed": convert_value_to_8bit_fixed(mode_intensity),
        "Brightness-2nd Mode_8bit_fixed": convert_value_to_8bit_fixed(mode2_intensity),
        "Brightness-Percentile_25th_8bit_fixed": convert_value_to_8bit_fixed(p25),
        "Brightness-Percentile_75th_8bit_fixed": convert_value_to_8bit_fixed(p75),
        "Gradient_90th_Percentile": calculate_gradient_extremes(img_array)
    }

# =====================================================================
# 2. 핵심 로직: 맵핑, 파일 처리, 모델 추론
# =====================================================================

PROFILE_MAPPING = {
    "Speckle_Reduction": ["SNR", "Speckle Index", "Homogeneity", "Non-Edge_CV"],
    "Contrast_Enhancement": ["RMS Contrast", "Contrast-Weber", "ST-Ten"],
    "Edge_Enhancement": ["Sharpness(Laplacian)", "Sharpness(Brenner)", "Connectivity-Num_Objects", "Connectivity-Avg_Size", "Connectivity-Largest_Ratio", "Gradient_X", "Gradient_Y"],
    "Gain_Increment": ["Brightness-Median_8bit_fixed", "Brightness-Mode_8bit_fixed", "Brightness-2nd Mode_8bit_fixed", "Brightness-Percentile_25th_8bit_fixed", "Brightness-Percentile_75th_8bit_fixed", "Gradient_90th_Percentile"]
}

def load_iq_min_max_info(file_path="iq_min_max_info.json"):
    """
    IQ Parameter Min&Max 정보 파일을 로드합니다.
    양식 예시: {"SNR": {"Min": 0.0, "Max": 100.0}, ...}
    *주의: 실제 운영 환경에 맞춰 파일명과 로드 방식을 변경하세요.*
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"경고: {file_path} 를 찾을 수 없습니다. 테스트용 더미 데이터를 반환합니다.")
        # 더미 데이터 생성 (모든 매핑된 키에 대해)
        dummy_data = {}
        for keys in PROFILE_MAPPING.values():
            for k in keys:
                dummy_data[k] = {"Min": 0.0, "Max": 100.0} 
        return dummy_data

def extract_width_height(raw_path):
    """경로나 파일명에서 _w(가로)_h(세로)_ 패턴을 찾아 추출합니다."""
    match = re.search(r'_w(\d+)_h(\d+)_', raw_path)
    if not match:
        raise ValueError(f"경로에서 width/height를 찾을 수 없습니다: {raw_path}")
    return int(match.group(1)), int(match.group(2))

def load_raw_image(raw_path):
    width, height = extract_width_height(raw_path)
    with open(raw_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype='<u2')
    if raw_data.size != width * height:
        raise ValueError("파일 크기가 width * height 와 일치하지 않습니다.")
    return raw_data.reshape((height, width))

def run_inference(iq_ref, iq_mod):
    """
    Siamese 모델과 Scaler를 로드하여 추론하는 부분입니다. 
    현재는 구조를 보여주기 위한 뼈대(Dummy 로직)로 구성되어 있습니다.
    """
    # 1. 모델 및 스케일러 로드 (경로는 환경에 맞게 수정)
    # model = torch.load("siamese_model.pth")
    # model.eval()
    # scaler = joblib.load("scaler.pkl")

    # 2. 피처 정렬 (모델 학습 시 사용한 IQ 파라미터 순서와 동일해야 함)
    feature_keys = list(iq_ref.keys())
    
    # x_ref = np.array([[iq_ref[k] for k in feature_keys]])
    # x_mod = np.array([[iq_mod[k] for k in feature_keys]])
    
    # 3. 스케일링
    # x_ref_scaled = scaler.transform(x_ref)
    # x_mod_scaled = scaler.transform(x_mod)
    
    # 4. 텐서 변환 및 추론
    # with torch.no_grad():
    #     t_ref = torch.tensor(x_ref_scaled, dtype=torch.float32)
    #     t_mod = torch.tensor(x_mod_scaled, dtype=torch.float32)
    #     pred_delta_style = model(t_ref, t_mod).numpy()[0]
    
    # ==== 더미 예측 로직 (실제 모델 연동 후 삭제) ====
    style_output_keys = [
        "Edge Threshold-Lv2", "Edge Threshold-Lv3", "LapSmoothRate-Lv2",
        "Adaptive Edge Smooth", "Adaptive Edge Contrast", "DecompositionType",
        "LimitationType", "DirPosition-Lv2"
    ]
    pred_delta_style = np.random.uniform(-1, 1, len(style_output_keys))
    # ================================================

    # 5. 결과 Json 포맷 맵핑
    result_json = {key: round(float(val), 4) for key, val in zip(style_output_keys, pred_delta_style)}
    return result_json

# =====================================================================
# 3. 메인 프로세스
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="IQ Parameter Inference Tool")
    parser.add_argument("input_json_path", help="4개의 파라미터 값이 들어있는 Json 파일 경로")
    parser.add_argument("rawdata_path", help="Raw 이미지 파일 경로 (파일명 또는 경로에 _w000_h000_ 포함 필수)")
    parser.add_argument("output_json_path", nargs="?", default="predicted_delta_style.json", help="추론 결과 저장 경로 (선택)")
    
    args = parser.parse_args()

    # 1. Json 파일 로드 > 4개의 profile 추출
    print(f"[1/6] Profile JSON 로드 중: {args.input_json_path}")
    with open(args.input_json_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    # 2. IQ 파라미터 Mapping 및 Δiq 계산
    print("[2/6] Min/Max 로드 및 Δiq(Delta IQ) 산출 중...")
    iq_info = load_iq_min_max_info() # 필요시 파일 경로 인자로 수정
    
    delta_iq = {}
    for profile_name, metric_list in PROFILE_MAPPING.items():
        prof_value = profiles.get(profile_name, 0) # e.g. -2, 5
        
        for metric in metric_list:
            if metric in iq_info:
                min_val = iq_info[metric]["Min"]
                max_val = iq_info[metric]["Max"]
                
                # 1Step = (Max-Min)/5
                step_val = (max_val - min_val) / 5.0
                
                # Δiq = 1Step * profile value
                delta_iq[metric] = step_val * prof_value
            else:
                delta_iq[metric] = 0.0

    # 3. Raw 파일 로드 > IQ 파라미터(IQ_ref) 계산
    print(f"[3/6] Raw 이미지 분석 중: {args.rawdata_path}")
    image_16bit = load_raw_image(args.rawdata_path)
    iq_ref = extract_base_iq_metrics(image_16bit)

    # 5. IQ_ref 와 Δiq 더함. 최종 IQ(IQ_mod) 산출 (요구사항 4번은 2번에 포함됨)
    print("[4&5/6] IQ_mod 산출 중...")
    iq_mod = {}
    for metric, ref_val in iq_ref.items():
        # 매핑된 파라미터에만 delta 적용
        _delta = delta_iq.get(metric, 0.0)
        iq_mod[metric] = ref_val + _delta

    # 6. 추론(Inference) 및 저장
    print("[6/6] Siamese 모델 추론 중...")
    output_data = run_inference(iq_ref, iq_mod)
    
    with open(args.output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"완료! 추론 결과가 저장되었습니다: {args.output_json_path}")

if __name__ == "__main__":
    main()