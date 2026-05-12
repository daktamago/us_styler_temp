import torch
import torch.nn as nn

class DynamicSiameseMultiTask(nn.Module):
    def __init__(self, input_dim, hidden_dims, reg_dim, cls_num_classes_list, dropout_rate=0.2):
        super(DynamicSiameseMultiTask, self).__init__()
        
        # 1. Shared Encoder (동적 생성)
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
        
        # 2. Base Feature Extractor (차이를 계산한 후 통과하는 네트워크)
        # 마지막 hidden_dim을 입력으로 받음
        feat_dim = hidden_dims[-1]
        self.feature_extractor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(feat_dim, feat_dim // 2), nn.BatchNorm1d(feat_dim // 2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        
        final_dim = feat_dim // 2
        
        # 3. Multi-Task Heads
        self.head_reg = nn.Linear(final_dim, reg_dim) if reg_dim > 0 else None
        
        if len(cls_num_classes_list) > 0:
            self.head_cls_list = nn.ModuleList([
                nn.Linear(final_dim, num_classes) for num_classes in cls_num_classes_list
            ])
        else:
            self.head_cls_list = None

    def forward(self, current_iq, target_iq):
        curr_feat = self.encoder(current_iq)
        tgt_feat = self.encoder(target_iq)
        
        # 차이 연산
        feat_diff = tgt_feat - curr_feat
        
        shared_features = self.feature_extractor(feat_diff)
        
        out_reg = self.head_reg(shared_features) if self.head_reg is not None else torch.empty(0).to(current_iq.device)
        
        out_cls = []
        if self.head_cls_list is not None:
            for cls_layer in self.head_cls_list:
                out_cls.append(cls_layer(shared_features))
                
        return out_reg, out_cls