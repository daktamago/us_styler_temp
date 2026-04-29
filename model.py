import torch
import torch.nn as nn

# [통합 아키텍처] Type 1, 3, 4, 5 전용 (Hybrid Multi-Task)
class SiameseStyleMultiTask_Base(nn.Module):
    def __init__(self, input_dim, reg_dim, cls_num_classes_list, dropout_rate=0.2):
        super(SiameseStyleMultiTask_Base, self).__init__()
        
        # 1. Shared Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        
        # 2. Base Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        
        # 3. Multi-Task Heads
        self.head_reg = nn.Linear(128, reg_dim) if reg_dim > 0 else None
        
        if len(cls_num_classes_list) > 0:
            self.head_cls_list = nn.ModuleList([
                nn.Linear(128, num_classes) for num_classes in cls_num_classes_list
            ])
        else:
            self.head_cls_list = None

    def forward(self, current_iq, target_iq):
        curr_feat = self.encoder(current_iq)
        tgt_feat = self.encoder(target_iq)
        feat_diff = tgt_feat - curr_feat
        
        shared_features = self.feature_extractor(feat_diff)
        
        out_reg = self.head_reg(shared_features) if self.head_reg is not None else torch.empty(0).to(current_iq.device)
        
        out_cls = []
        if self.head_cls_list is not None:
            for cls_layer in self.head_cls_list:
                out_cls.append(cls_layer(shared_features))
                
        return out_reg, out_cls