import torch
import torch.nn as nn

class SiameseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, extractor_dims, cls_head_dims, cls_num_list, dropout_rate=0.2):
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
        return [head(feat) for head in self.cls_heads]\n