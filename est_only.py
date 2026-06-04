import os
import re
import sys
import glob
import shutil
import subprocess  # 하위 스크립트 실행을 위해 추가
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from scipy import ndimage
from scipy.signal import convolve2d

# =========================================================
# [디렉토리 구성 정의]
# =========================================================
INPUT_RAW_DIR = "Input_Raw"
os.makedirs(INPUT_RAW_DIR, exist_ok=True)

# =========================================================
# 1. Siamese 딥러닝 모델 아키텍처 정의 (PyTorch)
# =========================================================
class SiameseRegressor(nn.Module):
    def __init__(self, input_dim=91, hidden_dims=[128, 64], extractor_dims=[32], reg_head_dims=[16], reg_dim=28, dropout_rate=0.2):
        super(SiameseRegressor, self).__init__()
        in_dim = input_dim
        enc_layers = []
        for h_dim in hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout_rate)])
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
       
        ext_layers = []
        curr_ext = hidden_dims[-1]
        if extractor_dims:
            for e_dim in extractor_dims:
                ext_layers.extend([nn.Linear(curr_ext, e_dim), nn.BatchNorm1d(e_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout_rate)])
                curr_ext = e_dim
        else:
            ext_layers.extend([nn.Linear(curr_ext, curr_ext), nn.BatchNorm1d(curr_ext), nn.LeakyReLU(0.01)])
        self.extractor = nn.Sequential(*ext_layers)
       
        head_layers = []
        curr_head = curr_ext
        if reg_head_dims:
            for r_dim in reg_head_dims:
                head_layers.extend([nn.Linear(curr_head, r_dim), nn.BatchNorm1d(r_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout_rate)])
                curr_head = r_dim
        head_layers.append(nn.Linear(curr_head, reg_dim))
        self.head_reg = nn.Sequential(*head_layers)

    def forward(self, curr_iq, tgt_iq):
        diff = self.encoder(tgt_iq) - self.encoder(curr_iq)
        return self.head_reg(self.extractor(diff))

class SiameseClassifier(nn.Module):
    def __init__(self, input_dim=91, hidden_dims=[128, 64], extractor_dims=[32], cls_head_dims=[16], cls_num_list=[9], dropout_rate=0.2):
        super(SiameseClassifier, self).__init__()
        in_dim = input_dim
        enc_layers = []
        for h_dim in hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout_rate)])
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
       
        ext_layers = []
        curr_ext = hidden_dims[-1]
        if extractor_dims:
            for e_dim in extractor_dims:
                ext_layers.extend([nn.Linear(curr_ext, e_dim), nn.BatchNorm1d(e_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout_rate)])
                curr_ext = e_dim
        else:
            ext_layers.extend([nn.Linear(curr_ext, curr_ext), nn.BatchNorm1d(curr_ext), nn.LeakyReLU(0.01)])
        self.extractor = nn.Sequential(*ext_layers)
       
        self.cls_heads = nn.ModuleList()
        for num_classes in cls_num_list:
            head_layers = []
            curr_head = curr_ext
            if cls_head_dims:
                for c_dim in cls_head_dims:
                    head_layers.extend([nn.Linear(curr_head, c_dim), nn.BatchNorm1d(c_dim), nn.LeakyReLU(0.01), nn.Dropout(dropout_rate)])
                    curr_head = c_dim
            head_layers.append(nn.Linear(curr_head, num_classes))
            self.cls_heads.append(nn.Sequential(*head_layers))

    def forward(self, curr_iq, tgt_iq):
        diff = self.encoder(tgt_iq) - self.encoder(curr_iq)
        feat = self.extractor(diff)
        return [head(feat) for head in self.cls_heads]

# =========================================================
# 2. IQ 파라미터 계산 핵심 알고리즘 정의
# =========================================================
def calculate_snr_with_speckle(image_16bit):
    if image_16bit is None or image_16bit.size == 0: return 0.0
    X = image_16bit.astype(np.float32)
    medianFiltered_image = ndimage.median_filter(X, size=5)
    deltaG_image = X - medianFiltered_image
    mean_medianFiltered_image = np.mean(medianFiltered_image)
    std_conventional = np.sqrt(np.mean(deltaG_image**2))
    if std_conventional < 1e-9: return 0.0
    snr_linear = mean_medianFiltered_image / std_conventional
    if snr_linear <= 0: return 0.0
    return 20 * np.log10(snr_linear)

def calculate_contrast_rms(image_16bit):
    return np.std(image_16bit, dtype=np.float64) if image_16bit.size > 0 else 0.0

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
    return float(np.median(image_16bit)) if image_16bit.size > 0 else 0.0

def calculate_mode_intensity(image_16bit):
    if image_16bit is None or image_16bit.size == 0: return 0.0, 0.0
    hist, bin_edges = np.histogram(image_16bit, bins=256, range=(0, 65536))
    mode_bin_idx = np.argmax(hist)
    mode_intensity = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2.0
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

def gaussian_downsample(image_16bit):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    blurred_image = convolve2d(image_16bit, kernel, mode='same', boundary='symm')
    return blurred_image[::2, ::2].astype(np.uint16)

# =========================================================
# 3. 전역 IQref 파라미터 필터링 추출
# =========================================================
def extract_full_iq_ref(image_16bit):
    image_lv1 = gaussian_downsample(image_16bit)
    image_lv2 = gaussian_downsample(image_lv1)
    image_lv3 = gaussian_downsample(image_lv2)
   
    images = {"": image_16bit, "_LV1": image_lv1, "_LV2": image_lv2, "_LV3": image_lv3}
    iq_dict = {}
   
    for suffix, img in images.items():
        mode_val, mode2_val = calculate_mode_intensity(img)
        p25, p75 = calculate_percentiles(img)
        conn_obj, conn_avg, conn_large = calculate_connectivity_auto_threshold(img)
       
        metrics = {
            f"SNR{suffix}": calculate_snr_with_speckle(img),
            f"RMS Contrast{suffix}": calculate_contrast_rms(img),
            f"Contrast-Weber{suffix}": calculate_weber_contrast(img),
            f"Sharpness(Laplacian){suffix}": calculate_sharpness_laplacian(img),
            f"Sharpness(Brenner){suffix}": calculate_sharpness_brenner(img),
            f"Speckle Index{suffix}": calculate_speckle_index(img),
            f"Homogeneity{suffix}": calculate_homogeneity(img),
            f"Brightness-Median_8bit_fixed{suffix}": convert_value_to_8bit_fixed(calculate_median_intensity(img)),
            f"Brightness-Mode_8bit_fixed{suffix}": convert_value_to_8bit_fixed(mode_val),
            f"Brightness-2nd Mode_8bit_fixed{suffix}": convert_value_to_8bit_fixed(mode2_val),
            f"Brightness-Percentile_25th_8bit_fixed{suffix}": convert_value_to_8bit_fixed(p25),
            f"Brightness-Percentile_75th_8bit_fixed{suffix}": convert_value_to_8bit_fixed(p75),
            f"Connectivity-Num_Objects{suffix}": conn_obj,
            f"Connectivity-Avg_Size{suffix}": conn_avg,
            f"Connectivity-Largest_Ratio{suffix}": conn_large
        }
           
        iq_dict.update(metrics)
       
    EXCLUDE_KEYS = [
        "Brightness-Mode_8bit_fixed",
        "Brightness-Mode_8bit_fixed_LV1",
        "Brightness-Percentile_25th_8bit_fixed_LV1",
        "Speckle Index_LV2",
        "Brightness-Mode_8bit_fixed_LV2",
        "Brightness-2nd Mode_8bit_fixed_LV3"
    ]
    for key in EXCLUDE_KEYS:
        if key in iq_dict:
            del iq_dict[key]
           
    return iq_dict

# =========================================================
# 4. 파라미터 타겟 정의
# =========================================================
CLASSIFIER_TARGETS = [
    "NR Strength-Lv2", "NR Strength-Lv3", "NR Curve Threshold-Lv0", "NR Curve Threshold-Lv1",
    "NR Curve Strength-Lv0", "NR Curve Strength-Lv1", "DecompositionType", "LimitationType", "DirPosition-Lv2"
]

REGRESSOR_TARGETS = [
    "Edge Threshold-Lv2", "Edge Threshold-Lv3", "LapSmoothRate-Lv2", "LapSmoothRate-Lv3", "LapAverageRate-Lv2", "LapAverageRate-Lv3",
    "Lap Gain Dark Area-Lv0", "Lap Gain Dark Area-Lv1", "Lap Gain Bright Area-Lv0", "Lap Gain Bright Area-Lv1", "Lap Gain Dark Area-Lv2",
    "Lap Gain Dark Area-Lv3", "Lap Gain Bright Area-Lv2", "Lap Gain Bright Area-Lv3", "Edge Smooth-Lv2", "Edge Smooth-Lv3", "Edge Contrast-Lv2",
    "Edge Contrast-Lv3", "Non-EdgeSmooth-Lv2", "Non-EdgeSmooth-Lv3", "Non-EdgeContrast-Lv2", "Non-EdgeContrast-Lv3", "DirPos Rate-Lv2", "DirPos Rate-Lv3",
    "DirNeg Rate-Lv2", "DirNeg Rate-Lv3", "Adaptive Edge Smooth", "Adaptive Edge Contrast"
]

# 사용자가 지정한 컬럼 순서
STYLE_COLUMNS_ORDER = [
    "Edge Threshold-Lv2", "Edge Threshold-Lv3", "LapSmoothRate-Lv2", "LapSmoothRate-Lv3",
    "LapAverageRate-Lv2", "LapAverageRate-Lv3", "NR Strength-Lv2", "NR Strength-Lv3",
    "NR Curve Threshold-Lv0", "NR Curve Threshold-Lv1", "NR Curve Strength-Lv0", "NR Curve Strength-Lv1",
    "Lap Gain Dark Area-Lv0", "Lap Gain Dark Area-Lv1", "Lap Gain Bright Area-Lv0", "Lap Gain Bright Area-Lv1",
    "Lap Gain Dark Area-Lv2", "Lap Gain Dark Area-Lv3", "Lap Gain Bright Area-Lv2", "Lap Gain Bright Area-Lv3",
    "Edge Smooth-Lv2", "Edge Smooth-Lv3", "Edge Contrast-Lv2", "Edge Contrast-Lv3",
    "Non-EdgeSmooth-Lv2", "Non-EdgeSmooth-Lv3", "Non-EdgeContrast-Lv2", "Non-EdgeContrast-Lv3",
    "DirPos Rate-Lv2", "DirPos Rate-Lv3", "DirNeg Rate-Lv2", "DirNeg Rate-Lv3",
    "Adaptive Edge Smooth", "Adaptive Edge Contrast", "DecompositionType", "LimitationType",
    "DirPosition-Lv2"
]

if __name__ == "__main__":
    print("=========================================================")
    print(" Style 및 IQ 파라미터 역정규화 보완 자동화 시스템 (컬럼 정렬 버전)")
    print("=========================================================")
   
    reg_model = SiameseRegressor(input_dim=54, hidden_dims=[512, 1024, 1024], extractor_dims=[1024, 512, 512], reg_head_dims=[512, 256], reg_dim=28)
    cls_model = SiameseClassifier(input_dim=54, hidden_dims=[512, 1024, 1024], extractor_dims=[512, 256], cls_head_dims=[256], cls_num_list=[17, 17, 21, 21, 17, 17, 3, 3, 3])
   
    try:
        reg_checkpoint = torch.load("model_Regressor.pth", map_location="cpu")
        if isinstance(reg_checkpoint, dict) and "model_state_dict" in reg_checkpoint:
            reg_model.load_state_dict(reg_checkpoint["model_state_dict"])
        else:
            reg_model.load_state_dict(reg_checkpoint)
        print(">> [성공] Regressor 모델 로드 완료.")
       
        cls_checkpoint = torch.load("model_Classifier.pth", map_location="cpu")
        if isinstance(cls_checkpoint, dict) and "model_state_dict" in cls_checkpoint:
            cls_model.load_state_dict(cls_checkpoint["model_state_dict"])
        else:
            cls_model.load_state_dict(cls_checkpoint)
        print(">> [성공] Classifier 모델 로드 완료.")
       
        reg_model.eval()
        cls_model.eval()
    except Exception as e:
        print(f">> [오류] 모델 로드 실패: {e}")
        sys.exit(1)
       
    minmax_file_path = "ParameterMinMaxStep_250613.csv"
    if not os.path.exists(minmax_file_path):
        print(f"[오류] {minmax_file_path} 파일이 존재하지 않습니다.")
        sys.exit(0)
       
    df_minmax = pd.read_csv(minmax_file_path, index_col=0)
    df_minmax.index = df_minmax.index.astype(str).str.strip().str.upper()
   
    scaler_path = "scaler_x.pkl"
    if not os.path.exists(scaler_path):
        if os.path.exists("scaler_X.pkl"): scaler_path = "scaler_X.pkl"
        else:
            print(f"[오류] 스케일러 파일이 존재하지 않습니다.")
            sys.exit(0)
    scaler = joblib.load(scaler_path)
    print(f">> 스케일러 파일({scaler_path}) 로드 완료.")
   
    print("\n[선택] Mod 파라미터 선택 방식을 결정하십시오 (1: 임의추출, 2: 수동입력)")
    choice = input("선택 번호: ").strip()
    mod_params = {}
    if choice == "1":
        if os.path.exists("IQ_List.csv"):
            df_list = pd.read_csv("IQ_List.csv")
            selected_df = df_list.sample(n=1)
            mod_params = selected_df.iloc[0].to_dict()
            selected_df.to_csv("selected_mod_params.csv", index=False)
            print(">> IQ_List.csv에서 성공적으로 1개 세트를 임의 추출하였습니다.")
        else:
            df_list = pd.read_csv("input_file_1.csv")
            selected_df = df_list.iloc[[0]]
            mod_params = selected_df.iloc[0].to_dict()
            selected_df.to_csv("selected_mod_params.csv", index=False)
    else:
        manual_input = input("수동 입력값: ").strip()
        if not manual_input:
            df_list = pd.read_csv("input_file_1.csv")
            selected_df = df_list.iloc[[0]]
            mod_params = selected_df.iloc[0].to_dict()
            selected_df.to_csv("selected_mod_params.csv", index=False)
        else:
            input_vals = [float(v) for v in manual_input.split(',')]
            df_list = pd.read_csv("input_file_1.csv")
            mod_params = dict(zip(df_list.columns, input_vals))
            pd.DataFrame([mod_params]).to_csv("selected_mod_params.csv", index=False)
           
    raw_files = glob.glob(os.path.join(INPUT_RAW_DIR, "*.raw"))
    if not raw_files:
        print(">> 처리 대상 raw 파일이 없습니다.")
        sys.exit(0)
       
    predicted_styles_data = []
   
    for raw_path in raw_files:
        filename = os.path.basename(raw_path)
        print(f"\n▶ 파일 분석 시작: {filename}")
       
        w_match = re.search(r'w(\d+)', filename)
        h_match = re.search(r'h(\d+)', filename)
        width = int(w_match.group(1)) if w_match else 720
        height = int(h_match.group(1)) if h_match else 249
       
        style_num = 0
        style_match = re.search(r'style(\d+)\.raw$', filename)
        if style_match: style_num = int(style_match.group(1))
       
        try:
            raw_img = np.fromfile(raw_path, dtype=np.uint16).reshape((height, width))
        except Exception as e:
            print(f"   - [에러] 파일 로드 실패: {e}")
            continue
           
        ref_iq = extract_full_iq_ref(raw_img)
       
        style_ref = {col: 0.0 for col in CLASSIFIER_TARGETS + REGRESSOR_TARGETS}
        if style_num != 0 and os.path.exists("Style_dict_37_re.csv"):
            df_dict = pd.read_csv("Style_dict_37_re.csv")
            matched_row = df_dict[df_dict['style'] == style_num]
            if not matched_row.empty:
                for col in style_ref.keys():
                    if col in matched_row.columns:
                        style_ref[col] = float(matched_row.iloc[0][col])
                       
        delta_style_normalized = {}
        try:
            iq_keys = list(ref_iq.keys())
            curr_iq_vec = np.array([ref_iq[k] for k in iq_keys], dtype=np.float32)
            tgt_iq_vec = np.array([mod_params.get(k, 0.0) for k in iq_keys], dtype=np.float32)
           
            curr_scaled = scaler.transform(curr_iq_vec.reshape(1, -1)).astype(np.float32)
            tgt_scaled = scaler.transform(tgt_iq_vec.reshape(1, -1)).astype(np.float32)
            curr_tensor = torch.tensor(curr_scaled)
            tgt_tensor = torch.tensor(tgt_scaled)
           
            with torch.no_grad():
                pred_reg = reg_model(tgt_tensor, curr_tensor).squeeze(0).numpy()
                pred_cls_logits = cls_model(tgt_tensor, curr_tensor)
                pred_cls = np.array([torch.argmax(logits, dim=1).item() for logits in pred_cls_logits])
               
            for idx, col in enumerate(REGRESSOR_TARGETS):
                delta_style_normalized[col] = float(pred_reg[idx])
            for idx, col in enumerate(CLASSIFIER_TARGETS):
                num_cls = cls_model.cls_heads[idx][-1].out_features if hasattr(cls_model, 'cls_heads') else 17
                pred_c = float(pred_cls[idx])
                delta_style_normalized[col] = ((pred_c / (num_cls - 1)) * 2.0 - 1.0) if num_cls > 1 else 0.0
               
        except Exception as e:
            print(f"   - [에러] 모델 추론 중 오류 발생: {e}")
            sys.exit(1)
           
        delta_style_restored = {}
        for col in CLASSIFIER_TARGETS + REGRESSOR_TARGETS:
            if col in df_minmax.columns:
                max_val = float(df_minmax.loc['MAX', col])
                min_val = float(df_minmax.loc['MIN', col])
                range_val = max_val - min_val
                delta_style_restored[col] = delta_style_normalized[col] * range_val
            else:
                delta_style_restored[col] = delta_style_normalized[col]
               
        final_style = {}
        for col in style_ref.keys():
            final_style[col] = style_ref[col] + delta_style_restored[col]
           
        predicted_styles_data.append({
            "FileName": filename,
            "StyleNum": style_num,
            "Type": "predicted_delta_normalized",
            **delta_style_normalized
        })
        predicted_styles_data.append({
            "FileName": filename,
            "StyleNum": style_num,
            "Type": "predicted_delta_restored",
            **delta_style_restored
        })
        predicted_styles_data.append({
            "FileName": filename,
            "StyleNum": style_num,
            "Type": "predicted_final_style",
            **final_style
        })
        print("   - [성공] Style Difference 예측 완료.")
       
    if predicted_styles_data:
        style_report_df = pd.DataFrame(predicted_styles_data)
       
        # 사용자가 지정한 컬럼 순서로 정렬 (ID 컬럼 포함)
        style_cols = ["FileName", "StyleNum", "Type"] + STYLE_COLUMNS_ORDER
        style_report_df = style_report_df[style_cols]
        style_report_df.to_csv("Predicted_Style_Parameters_Report.csv", index=False)
        print(">> [결과 저장 완료] Predicted_Style_Parameters_Report.csv")
       
        # de-normalized style difference만 따로 저장 (순서 동일 적용)
        style_diff_only_df = style_report_df[style_report_df["Type"] == "predicted_delta_restored"].copy()
        style_diff_only_df = style_diff_only_df.drop(columns=["Type"])
       
        # 파일명, Style 번호, 그리고 37개 파라미터가 지정된 순서대로 정렬되어 저장됩니다.
        style_diff_cols = ["FileName", "StyleNum"] + STYLE_COLUMNS_ORDER
        style_diff_only_df = style_diff_only_df[style_diff_cols]
       
        style_diff_only_df.to_csv("Predicted_Style_Difference.csv", index=False)
        print(">> [결과 저장 완료] Predicted_Style_Difference.csv")

        # [추가된 기능] Predicted_Style_Difference.csv 와 Style_dict_37.csv 를 사용하여 Final_Style.csv 생성
        style_dict_path = "Style_dict_37.csv"
        if os.path.exists(style_dict_path):
            # 2. Style_dict_37.csv에서 [StyleNum]의 숫자를 검색하기 위해 인덱스로 설정
            style_dict_df = pd.read_csv(style_dict_path)
            style_dict_indexed = style_dict_df.set_index('style')
           
            # 1. 양식은 Predicted_Style_Difference.csv (현재 style_diff_only_df) 와 동일하게 구성
            final_style_df = style_diff_only_df.copy()
            params_cols = final_style_df.columns[2:] # 3번째 Column부터 마지막 Column까지
           
            for idx, row in final_style_df.iterrows():
                style_num = row['StyleNum']
                if style_num in style_dict_indexed.index:
                    # 2. 일치하는 Row의 값을 복사하여 NS 값으로 명명
                    ns_values = style_dict_indexed.loc[style_num]
                   
                    # 3. NS 값과 Predicted_Style_Difference.csv 의 파라미터 값을 Column by Column 으로 더함
                    final_style_df.loc[idx, params_cols] = row[params_cols] + ns_values[params_cols].values
                   
            # 4. & 5. 최종 결과값을 동일한 양식으로 기록 및 Final_Style.csv로 저장
            final_style_df.to_csv("Final_Style.csv", index=False)
            print(">> [결과 저장 완료] Final_Style.csv")
            
            # =========================================================
            # [신규 기능 추가부] 기능 1, 2, 3 구현
            # =========================================================
            # 1. Final_Style.csv 에서 [StyleNum] 컬럼부터 끝까지 슬라이싱하여 parameter_cleaned.csv 생성 (헤더 없음)
            cleaned_df = final_style_df.loc[:, "StyleNum":]
            cleaned_file_path = "parameter_cleaned.csv"
            cleaned_df.to_csv(cleaned_file_path, header=False, index=False)
            print(">> [결과 저장 완료] parameter_cleaned.csv (헤더 제외)")
            
            # 2. 하위폴더 RSDMR_generator 에 parameter_cleaned.csv 파일 복사 (덮어쓰기)
            target_dir = "RSDMR_generator"
            if os.path.exists(target_dir):
                target_file_path = os.path.join(target_dir, cleaned_file_path)
                shutil.copy2(cleaned_file_path, target_file_path)
                print(f">> [파일 복사 완료] {cleaned_file_path} -> {target_file_path}")
                
                # 3. 하위폴더 RSDMR_generator의 Generate_RSDMR.py 실행
                script_name = "Generate_RSDMR.py"
                script_path = os.path.join(target_dir, script_name)
                if os.path.exists(script_path):
                    print(f">> [스크립트 실행] {script_name} 구동 시작...")
                    try:
                        # 하위 폴더 내부 환경에서 스크립트가 에러 없이 실행되도록 cwd(작업 디렉토리)를 변경하여 실행
                        result = subprocess.run([sys.executable, script_name], cwd=target_dir, check=True)
                        print(f">> [성공] {script_name} 실행이 정상 종료되었습니다.")
                    except subprocess.CalledProcessError as e:
                        print(f">> [오류] {script_name} 실행 중 문제가 발생했습니다: {e}")
                else:
                    print(f">> [오류] {script_path} 파일이 존재하지 않아 스크립트를 실행할 수 없습니다.")
            else:
                print(f">> [오류] 하위 폴더 {target_dir} 이 존재하지 않아 복사 및 실행을 중단합니다.")
            # =========================================================
            
        else:
            print(f">> [안내] {style_dict_path} 파일이 존재하지 않아 Final_Style.csv를 생성할 수 없습니다.")

       
    print("\n=========================================================")
    print(" 모든 이미지 처리가 완료되었습니다.")
    print("=========================================================")