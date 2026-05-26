# model.py
import torch
import torch.nn as nn
import pandas as pd
import os

class TabularStyleTransformer(nn.Module):
    def __init__(self, class_indices, reg_indices, config):
        super().__init__()
        
        d_model = config['D_MODEL']
        self.iq_group_size = config['IQ_GROUP_SIZE']
        self.num_iq_groups = config['IQ_PARAM_COUNT'] // self.iq_group_size
        self.num_style_params = config['STYLE_PARAM_COUNT']
        
        assert config['IQ_PARAM_COUNT'] % self.iq_group_size == 0, "IQ 개수는 Group Size로 나누어 떨어져야 합니다."
        
        self.register_buffer('class_idx', torch.tensor(class_indices, dtype=torch.long))
        self.register_buffer('reg_idx', torch.tensor(reg_indices, dtype=torch.long))
        
        # 1. 동적 IQ Embedding
        self.iq_embedding = nn.Linear(self.iq_group_size, d_model)
        
        # 2. Self-Attention Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=config['N_HEADS'], batch_first=True)
        self.iq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['NUM_LAYERS'])
        
        # 3. Learnable Queries
        self.style_queries = nn.Parameter(torch.randn(self.num_style_params, d_model))
        
        # 4. Cross-Attention Decoder
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=config['N_HEADS'], batch_first=True)
        
        # 5. Output Heads (분류기 최대 클래스 개수 동적 적용)
        self.regressor_head = nn.Linear(d_model, 1)
        self.classifier_head = nn.Linear(d_model, config['MAX_CLASSES']) 

    def forward(self, delta_iq):
        batch_size = delta_iq.size(0)
        
        # 동적 Grouping 연산
        grouped_iq = delta_iq.view(batch_size, self.num_iq_groups, self.iq_group_size)
        iq_emb = self.iq_embedding(grouped_iq)
        iq_encoded = self.iq_encoder(iq_emb)
        
        q = self.style_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attn_output, attn_weights = self.cross_attention(query=q, key=iq_encoded, value=iq_encoded, need_weights=True)
        
        class_tokens = attn_output[:, self.class_idx, :]
        reg_tokens = attn_output[:, self.reg_idx, :]
        
        pred_class = self.classifier_head(class_tokens) 
        pred_reg = self.regressor_head(reg_tokens).squeeze(-1) 
        
        return pred_class, pred_reg, attn_weights

    @torch.no_grad()
    def save_explainable_artifacts(self, style_names, save_dir="artifacts"):
        os.makedirs(save_dir, exist_ok=True)
        
        embed_weights = self.iq_embedding.weight.detach().cpu().numpy()
        pd.DataFrame(embed_weights, columns=[f'Lv{i}' for i in range(self.iq_group_size)]).to_csv(
            os.path.join(save_dir, 'iq_scale_embedding_filter.csv'), index=False
        )
        
        queries = self.style_queries.detach().cpu()
        queries_norm = torch.nn.functional.normalize(queries, p=2, dim=1)
        correlation_matrix = torch.mm(queries_norm, queries_norm.t()).numpy()
        pd.DataFrame(correlation_matrix, index=style_names, columns=style_names).to_csv(
            os.path.join(save_dir, 'style_parameter_correlation.csv')
        )