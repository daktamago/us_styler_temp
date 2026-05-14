import torch
import torch.nn as nn

class SiameseRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, extractor_dims, reg_head_dims, reg_dim, dropout_rate=0.2):
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
        return self.head_reg(self.extractor(diff))\n