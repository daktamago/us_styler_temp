import cv2
import os
import re
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor, as_completed

### Existing Image Quality Factor Calculation Functions ###

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

def gaussian_downsample_scipy(image_16bit):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    blurred_image = convolve2d(image_16bit, kernel, mode='same', boundary='symm')
    return blurred_image[::2, ::2].astype(np.uint16)


### NEW Image Quality Factor Calculation Functions ###

def calculate_directional_gradients(image_16bit):
    """X축(수평) 및 Y축(수직) 방향의 그래디언트 강도를 분리하여 계산합니다."""
    if image_16bit is None or image_16bit.size == 0: return 0.0, 0.0
    img_float = image_16bit.astype(np.float32)
    sobel_x = ndimage.sobel(img_float, axis=1)
    sobel_y = ndimage.sobel(img_float, axis=0)
    return float(np.mean(np.abs(sobel_x))), float(np.mean(np.abs(sobel_y)))

def calculate_gradient_extremes(image_16bit, percentile=90):
    """상위 N%에 해당하는 가장 날카로운 엣지의 강도를 계산합니다."""
    if image_16bit is None or image_16bit.size == 0: return 0.0
    img_float = image_16bit.astype(np.float32)
    sobel_x = ndimage.sobel(img_float, axis=1)
    sobel_y = ndimage.sobel(img_float, axis=0)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.percentile(grad_mag, percentile))

def calculate_masked_cv(image_16bit, bottom_percentile=20):
    """엣지가 없는 평탄부(Non-Edge) 영역만의 노이즈/질감(CV)을 계산합니다."""
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
    """노이즈를 억제하고 임계값(tau) 이상의 유의미한 구조적 엣지만 평가합니다 (논문 기반)."""
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


def process_single_file(file_path, width, height, df_params):
    def extract_iq_metrics(img_array, suffix=""):
        # 기존 지표
        snr = calculate_snr_with_speckle(img_array)
        rms_contrast = calculate_contrast_rms(img_array)
        weber_contrast = calculate_weber_contrast(img_array)
        sharpness_lap = calculate_sharpness_laplacian(img_array)
        sharpness_bren = calculate_sharpness_brenner(img_array)
        speckle_idx = calculate_speckle_index(img_array)
        homogeneity = calculate_homogeneity(img_array)
        
        median_intensity = calculate_median_intensity(img_array)
        mode_intensity, mode2_intensity = calculate_mode_intensity(img_array)
        p25, p75 = calculate_percentiles(img_array)
        
        median_8bit = convert_value_to_8bit_fixed(median_intensity)
        mode_8bit = convert_value_to_8bit_fixed(mode_intensity)
        mode2_8bit = convert_value_to_8bit_fixed(mode2_intensity)
        p25_8bit = convert_value_to_8bit_fixed(p25)
        p75_8bit = convert_value_to_8bit_fixed(p75)
        
        num_objects, avg_object_size, largest_object_ratio = calculate_connectivity_auto_threshold(img_array)

        # 신규 지표 계산
        grad_x, grad_y = calculate_directional_gradients(img_array)
        grad_90th = calculate_gradient_extremes(img_array, percentile=90)
        masked_cv = calculate_masked_cv(img_array, bottom_percentile=20)
        st_ten = calculate_st_ten(img_array)
        
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
            f"Connectivity-Num_Objects{suffix}": num_objects,
            f"Connectivity-Avg_Size{suffix}": avg_object_size,
            f"Connectivity-Largest_Ratio{suffix}": largest_object_ratio,
            
            # 신규 5개 지표 병합
            f"Gradient_X{suffix}": grad_x,
            f"Gradient_Y{suffix}": grad_y,
            f"Gradient_90th_Percentile{suffix}": grad_90th,
            f"Non-Edge_CV{suffix}": masked_cv,
            f"ST-Ten{suffix}": st_ten
        }

    try:
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype='<u2')
        
        if raw_data.size != width * height:
            return None
            
        image_16bit = raw_data.reshape((height, width))
        image_lv1 = gaussian_downsample_scipy(image_16bit)
        image_lv2 = gaussian_downsample_scipy(image_lv1)
        image_lv3 = gaussian_downsample_scipy(image_lv2)
        
        quality_metrics = {"File Path": file_path}
        quality_metrics.update(extract_iq_metrics(image_16bit, suffix=""))
        quality_metrics.update(extract_iq_metrics(image_lv1, suffix="_LV1"))
        quality_metrics.update(extract_iq_metrics(image_lv2, suffix="_LV2"))
        quality_metrics.update(extract_iq_metrics(image_lv3, suffix="_LV3"))
        
        filename = os.path.basename(file_path)
        style_match = re.search(r'style_(\d+)', filename, re.IGNORECASE)
        
        param_values = {}
        if style_match:
            style_number = int(style_match.group(1))
            try:
                param_series = df_params.loc[style_number]
                param_values = param_series.to_dict()
            except KeyError:
                pass
        
        return {**quality_metrics, **param_values}
        
    except Exception as e:
        return None

def process_folders(root_path, param_file_path, output_file):
    df_params = pd.read_excel(param_file_path).set_index('style')
    results, tasks = [], []
    
    for dirpath, _, filenames in os.walk(root_path):
        if dirpath == r'C:\workspace_medical\ValidationData\260223_StyleUS_forTest\test_data\3VV PA\SDMR_Input_w1008_h394_14_36_14_338':
            continue

        match = re.search(r'_w(\d+)_h(\d+)_', os.path.basename(dirpath))
        if not match: continue
        width, height = int(match.group(1)), int(match.group(2))
        
        for filename in filenames:
            if filename.lower().endswith('.raw') and 'style' in filename.lower():
                tasks.append((os.path.join(dirpath, filename), width, height, df_params))
    
    if not tasks: return

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, *task): task[0] for task in tasks}
        for future in as_completed(future_to_file):
            if result := future.result(): results.append(result)

    if not results: return
        
    df_results = pd.DataFrame(results).sort_values(by="File Path").reset_index(drop=True)
    
    # 신규 지표들이 포함된 베이스 메트릭 리스트 확장 (기존 15개 + 신규 5개 = 총 20개)
    base_metrics = [
        "SNR", "RMS Contrast", "Contrast-Weber", "Sharpness(Laplacian)", 
        "Sharpness(Brenner)", "Speckle Index", "Homogeneity", 
        "Brightness-Median_8bit_fixed", "Brightness-Mode_8bit_fixed", 
        "Brightness-2nd Mode_8bit_fixed", "Brightness-Percentile_25th_8bit_fixed", 
        "Brightness-Percentile_75th_8bit_fixed", "Connectivity-Num_Objects",
        "Connectivity-Avg_Size", "Connectivity-Largest_Ratio",
        "Gradient_X", "Gradient_Y", "Gradient_90th_Percentile", 
        "Non-Edge_CV", "ST-Ten"
    ]
    
    suffixes = ["", "_LV1", "_LV2", "_LV3"]
    quality_cols = ["File Path"]
    for suffix in suffixes:
        for metric in base_metrics:
            quality_cols.append(f"{metric}{suffix}")
    
    param_cols = df_params.columns.tolist()
    final_cols_order = [col for col in quality_cols if col in df_results.columns] + \
                       [col for col in param_cols if col in df_results.columns]
    
    df_results[final_cols_order].to_excel(output_file, index=False, engine='openpyxl')

if __name__ == '__main__':
    ROOT_FOLDER_PATH = r"C:\workspace_medical\ValidationData\260223_StyleUS_forTest\test_data"
    PARAMS_EXCEL_FILE = "Internal_Params_RZ20_Style1-7488.xlsx"
    OUTPUT_EXCEL_FILE = "image_quality_metrics_with_new_params_pyramidal_val.xlsx"
    
    process_folders(ROOT_FOLDER_PATH, PARAMS_EXCEL_FILE, OUTPUT_EXCEL_FILE)