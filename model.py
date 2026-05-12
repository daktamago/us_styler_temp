import torch
import torch.nn as nn

class DynamicSiameseMultiTask(nn.Module):
    def __init__(self, input_dim, hidden_dims, reg_head_dims, cls_head_dims, reg_dim, cls_num_classes_list, dropout_rate=0.2):
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
        
        # 2. Base Feature Extractor (Difference Processing)
        extractor_layers = []
        curr_ext_in = hidden_dims[-1] # Encoder의 마지막 출력
        
        if extractor_dims:
            for e_dim in extractor_dims:
                extractor_layers.extend([
                    nn.Linear(curr_ext_in, e_dim),
                    nn.BatchNorm1d(e_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate)
                ])
                curr_ext_in = e_dim
        else:
            # 입력값이 없으면 차원 유지 (Bottleneck 방지)
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
            for h_dim in head_hidden_list:
                layers.extend([
                    nn.Linear(curr_in, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate)
                ])
                curr_in = h_dim
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
        curr_feat = self.encoder(current_iq)
        tgt_feat = self.encoder(target_iq)
        
        # Siamese interaction: Difference
        feat_diff = tgt_feat - curr_feat
        
        shared_features = self.feature_extractor(feat_diff)
        
        out_reg = self.head_reg(shared_features) if self.head_reg is not None else torch.empty(0).to(current_iq.device)
        
        out_cls = []
        if self.head_cls_list is not None:
            for cls_layer in self.head_cls_list:
                out_cls.append(cls_layer(shared_features))
                
        return out_reg, out_cls