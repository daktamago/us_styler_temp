import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StyleDifferenceDataset

def run_multitask_training_pipeline(model, X_train, y_train, X_val, y_val, 
                                    reg_indices, cls_indices, cls_num_classes_list, style_names,
                                    custom_weights=None, # 🔥 동적 가중치 딕셔너리 받기
                                    batch_size=256, epochs=50, lr=1e-3, device='cuda'):
    
    if custom_weights is None:
        custom_weights = {}

    print(f"\n🚀 [Multi-Task Hybrid] 학습 시작...")
    
    train_loader = DataLoader(StyleDifferenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    
    criterion_reg = nn.HuberLoss(reduction='none') 
    criterion_cls = nn.CrossEntropyLoss(reduction='none')
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 📌 동적 가중치 추출 헬퍼 함수
    def get_dynamic_weight(param_name):
        name_upper = param_name.upper()
        weight = 1.0
        # 설정된 키워드가 파라미터명에 포함되어 있으면 해당 가중치 부여 (중복 시 가장 큰 가중치 적용)
        for keyword, w in custom_weights.items():
            if keyword in name_upper:
                weight = max(weight, w) 
        return weight

    # 🔥 Regression 파라미터 동적 가중치 텐서 생성
    reg_weights = []
    for idx in reg_indices:
        name = style_names[idx]
        reg_weights.append(get_dynamic_weight(name))
        
    reg_weights_tensor = torch.tensor(reg_weights, dtype=torch.float32).to(device) if reg_indices else None

    for epoch in range(epochs):
        model.train()
        train_loss_total, train_loss_reg, train_loss_cls = 0.0, 0.0, 0.0
        
        for curr_iq, tgt_iq, actual_diff in train_loader:
            curr_iq, tgt_iq, actual_diff = curr_iq.to(device), tgt_iq.to(device), actual_diff.to(device)
            
            diff_reg = actual_diff[:, reg_indices] if len(reg_indices) > 0 else None
            diff_cls = actual_diff[:, cls_indices] if len(cls_indices) > 0 else None
            
            optimizer.zero_grad()
            pred_reg, pred_cls_list = model(curr_iq, tgt_iq)
            loss = 0.0
            
            # [A] Regression Loss
            if diff_reg is not None and diff_reg.numel() > 0:
                raw_l_reg = criterion_reg(pred_reg, diff_reg) # (Batch, Reg_Dim)
                weighted_l_reg = raw_l_reg * reg_weights_tensor
                l_reg = weighted_l_reg.mean()
                loss += l_reg
                train_loss_reg += l_reg.item()
                
            # [B] Classification Loss
            if diff_cls is not None and diff_cls.numel() > 0:
                l_cls_total = 0.0
                for i, num_classes in enumerate(cls_num_classes_list):
                    target_val = diff_cls[:, i]
                    target_idx = torch.round((target_val + 1.0) / 2.0 * (num_classes - 1)).long()
                    target_idx = torch.clamp(target_idx, 0, num_classes - 1)
                    
                    raw_l_cls = criterion_cls(pred_cls_list[i], target_idx) # (Batch,)
                    
                    # 🔥 Classification 파라미터 동적 가중치 할당
                    cls_param_name = style_names[cls_indices[i]]
                    weight = get_dynamic_weight(cls_param_name)
                    
                    l_cls = raw_l_cls.mean() * weight
                    l_cls_total += l_cls
                
                l_cls_total = l_cls_total / len(cls_num_classes_list)
                loss += l_cls_total
                train_loss_cls += l_cls_total.item()
            
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            avg_tot = train_loss_total / len(train_loader)
            print(f"   Epoch [{epoch+1}/{epochs}] Total Loss: {avg_tot:.4f}")
            
    return model