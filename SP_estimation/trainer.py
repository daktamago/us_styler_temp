import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StyleDirectDataset # 수정된 데이터셋

def run_multitask_training_pipeline(model, X_train, y_train, X_val, y_val, 
                                    reg_indices, cls_indices, cls_num_classes_list, style_names,
                                    custom_weights=None, batch_size=256, epochs=50, lr=1e-3, device='cuda'):
    
    if custom_weights is None: custom_weights = {}

    train_loader = DataLoader(StyleDirectDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(StyleDirectDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    criterion_reg = nn.HuberLoss(reduction='none') 
    criterion_cls = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    def get_dynamic_weight(param_name):
        name_upper = param_name.upper()
        weight = 1.0
        for keyword, w in custom_weights.items():
            if keyword in name_upper: weight = max(weight, w) 
        return weight

    reg_weights_tensor = torch.tensor([get_dynamic_weight(style_names[idx]) for idx in reg_indices], dtype=torch.float32).to(device) if reg_indices else None

    for epoch in range(epochs):
        model.train()
        train_loss_total, train_loss_reg, train_loss_cls = 0.0, 0.0, 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            pred_reg, pred_cls_list = model(inputs) # 모델 입력 변경
            loss = 0.0
            
            if len(reg_indices) > 0:
                l_reg = (criterion_reg(pred_reg, targets[:, reg_indices]) * reg_weights_tensor).mean()
                loss += l_reg
                train_loss_reg += l_reg.item()
                
            if len(cls_indices) > 0:
                l_cls_batch = 0.0
                for i, num_classes in enumerate(cls_num_classes_list):
                    t_idx = torch.clamp(torch.round((targets[:, cls_indices[i]] + 1.0) / 2.0 * (num_classes - 1)).long(), 0, num_classes-1)
                    raw_l_cls = criterion_cls(pred_cls_list[i], t_idx)
                    dist_penalty = torch.abs(torch.argmax(pred_cls_list[i], dim=1) - t_idx).float()
                    weight = get_dynamic_weight(style_names[cls_indices[i]])
                    l_cls_batch += (raw_l_cls + 0.1 * dist_penalty).mean() * weight
                l_cls_batch /= len(cls_num_classes_list)
                loss += l_cls_batch
                train_loss_cls += l_cls_batch.item()
            
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for v_in, v_tgt in val_loader:
                    v_in, v_tgt = v_in.to(device), v_tgt.to(device)
                    vp_reg, vp_cls_list = model(v_in)
                    v_loss = 0.0
                    if len(reg_indices) > 0: v_loss += (criterion_reg(vp_reg, v_tgt[:, reg_indices]) * reg_weights_tensor).mean()
                    if len(cls_indices) > 0:
                        v_l_cls = 0.0
                        for i, num_classes in enumerate(cls_num_classes_list):
                            vt_idx = torch.clamp(torch.round((v_tgt[:, cls_indices[i]] + 1.0) / 2.0 * (num_classes - 1)).long(), 0, num_classes-1)
                            v_l_cls += (criterion_cls(vp_cls_list[i], vt_idx) + 0.1 * torch.abs(torch.argmax(vp_cls_list[i], dim=1) - vt_idx).float()).mean() * get_dynamic_weight(style_names[cls_indices[i]])
                        v_loss += (v_l_cls / len(cls_num_classes_list))
                    val_loss_total += v_loss.item()
            
            print(f"   Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss_total/len(train_loader):.4f} | Val Loss: {val_loss_total/len(val_loader):.4f}")
            
    return model