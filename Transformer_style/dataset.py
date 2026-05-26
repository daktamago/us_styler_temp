# dataset.py
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # 스케일러 추가

def get_parameter_routing_indices(df_columns, iq_count, classifier_names):
    """동적 라우팅을 위한 인덱스와 이름을 추출합니다."""
    style_columns = df_columns[iq_count:]
    class_indices = []
    reg_indices = []
    
    for idx, col_name in enumerate(style_columns):
        clean_name = col_name.strip()
        if clean_name in classifier_names:
            class_indices.append(idx)
        else:
            reg_indices.append(idx)
            
    return class_indices, reg_indices, list(style_columns)

class SiameseTabularDataset(Dataset):
    def __init__(self, x_data, y_data, num_pairs_per_epoch):
        super().__init__()
        self.num_pairs = num_pairs_per_epoch
        self.X = x_data.astype('float32')
        self.Y = y_data.astype('float32')
        self.data_size = len(self.X)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # 무작위로 두 개의 서로 다른 행(Row A, Row B) 추출
        idx_a, idx_b = np.random.choice(self.data_size, 2, replace=False)
        
        # 정규화가 완료된 입력값들 간의 (Target - Current) 차이값 연산
        delta_iq = self.X[idx_a] - self.X[idx_b]
        delta_style = self.Y[idx_a] - self.Y[idx_b]
        
        return torch.tensor(delta_iq), torch.tensor(delta_style)

def prepare_dataloaders(csv_path, config):
    df = pd.read_csv(csv_path)
    iq_count = config['IQ_PARAM_COUNT']
    style_count = config['STYLE_PARAM_COUNT']
    
    # 1. 이름 기반 동적 라우팅 인덱스 추출
    class_idx, reg_idx, style_names = get_parameter_routing_indices(
        df.columns, iq_count, config['CLASSIFIER_NAMES']
    )
    
    # 2. Train / Val Split (수치 변환 전 분할하여 Data Leakage 전면 방지)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 3. 데이터 분리 [IQ Parameter]와 [Style Parameter]
    train_x_raw = train_df.iloc[:, :iq_count].values
    train_y = train_df.iloc[:, iq_count:iq_count+style_count].values
    
    val_x_raw = val_df.iloc[:, :iq_count].values
    val_y = val_df.iloc[:, iq_count:iq_count+style_count].values
    
    # 4. 🛠️ IQ Parameter 전용 동적 스케일링 프로세스 추가
    # 오직 Train 데이터의 IQ 영역만 쳐다보고 가이드라인(Min/Max 값)을 생성합니다.
    # dataset.py 의 prepare_dataloaders 함수 하단 부분
    
    # ... (이전 코드 동일) ...
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_x_scaled = scaler.fit_transform(train_x_raw)
    val_x_scaled = scaler.transform(val_x_raw)
    
    train_dataset = SiameseTabularDataset(train_x_scaled, train_y, config['NUM_PAIRS_PER_EPOCH'])
    val_dataset = SiameseTabularDataset(val_x_scaled, val_y, int(config['NUM_PAIRS_PER_EPOCH'] * 0.2))
    
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    # 💡 수정됨: 마지막에 scaler 객체도 함께 반환합니다.
    return train_loader, val_loader, class_idx, reg_idx, style_names, scaler