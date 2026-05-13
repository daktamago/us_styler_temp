import torch
import torch.nn as nn

class DynamicSiameseMultiTask(nn.Module):
    # 여기에 extractor_dims가 포함되어 있어야 합니다.
    def __init__(self, input_dim, hidden_dims, extractor_dims, reg_head_dims, cls_head_dims, reg_dim, cls_num_classes_list, dropout_rate=0.2):
        super(DynamicSiameseMultiTask, self).__init__()
        
        # 1. Shared Encoder (IQ Feature Extraction)
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 2. Base Feature Extractor (Difference Processing / Subtractor)
        extractor_layers = []
        curr_ext_in = hidden_dims[-1] # 인코더의 마지막 차원을 입력으로 받음
        
        if extractor_dims: # 사용자가 차원을 입력한 경우
            for e_dim in extractor_dims:
                extractor_layers.extend([
                    nn.Linear(curr_ext_in, e_dim),
                    nn.BatchNorm1d(e_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate)
                ])
                curr_ext_in = e_dim
        else:
            # 입력값이 없으면 차원 축소(Bottleneck) 없이 그대로 유지
            extractor_layers.extend([
                nn.Linear(curr_ext_in, curr_ext_in),
                nn.BatchNorm1d(curr_ext_in),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            
        self.feature_extractor = nn.Sequential(*extractor_layers)
        final_shared_dim = curr_ext_in
        
        # Head 생성을 위한 헬퍼 함수
        def build_head(in_features, head_hidden_list, out_features):
            layers = []
            curr_in = in_features
            if head_hidden_list: # 리스트가 비어있지 않을 때만 히든 레이어 추가
                for h_dim in head_hidden_list:
                    layers.extend([
                        nn.Linear(curr_in, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_rate)
                    ])
                    curr_in = h_dim
            # 마지막 출력 레이어 (활성화 함수 없음)
            layers.append(nn.Linear(curr_in, out_features))
            return nn.Sequential(*layers)

        # 3. Multi-Task Heads (Independent Structure)
        self.head_reg = build_head(final_shared_dim, reg_head_dims, reg_dim) if reg_dim > 0 else None
        
        if len(cls_num_classes_list) > 0:
            self.head_cls_list = nn.ModuleList([
                build_head(final_shared_dim, cls_head_dims, num_classes) for num_classes in cls_num_classes_list
            ])
        else:
            self.head_cls_list = None

    def forward(self, current_iq, target_iq):
        # 샴 구조 연산 (두 입력을 각각 동일한 인코더에 통과)
        curr_feat = self.encoder(current_iq)
        tgt_feat = self.encoder(target_iq)
        
        # Siamese interaction: Difference 연산
        feat_diff = tgt_feat - curr_feat
        
        # 차이값을 추출기(Subtractor)에 통과
        shared_features = self.feature_extractor(feat_diff)
        
        # 각각의 Head로 분기하여 결과 도출
        out_reg = self.head_reg(shared_features) if self.head_reg is not None else torch.empty(0).to(current_iq.device)
        
        out_cls = []
        if self.head_cls_list is not None:
            for cls_layer in self.head_cls_list:
                out_cls.append(cls_layer(shared_features))
                
        return out_reg, out_cls