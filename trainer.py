import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StyleDifferenceDataset

def run_multitask_training_pipeline(model, X_train, y_train, X_val, y_val, 
                                    reg_indices, cls_indices, cls_num_classes_list, style_names,
                                    custom_weights=None, batch_size=256, epochs=50, lr=1e-3, device='cuda'):
    
    if custom_weights is None:
        custom_weights = {}

    print(f"\n[Multi-Task Hybrid] Training Initialized...")
    
    train_loader = DataLoader(StyleDifferenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(StyleDifferenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    criterion_reg = nn.HuberLoss(reduction='none') 
    criterion_cls = nn.CrossEntropyLoss(reduction='none')
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    def get_dynamic_weight(param_name):
        name_upper = param_name.upper()
        weight = 1.0
        for keyword, w in custom_weights.items():
            if keyword in name_upper:
                weight = max(weight, w) 
        return weight

    reg_weights_tensor = None
    if reg_indices:
        reg_weights = [get_dynamic_weight(style_names[idx]) for idx in reg_indices]
        reg_weights_tensor = torch.tensor(reg_weights, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        # --- [Training Phase] ---
        model.train()
        train_loss_total = 0.0
        
        for curr_iq, tgt_iq, actual_diff in train_loader:
            curr_iq, tgt_iq, actual_diff = curr_iq.to(device), tgt_iq.to(device), actual_diff.to(device)
            
            optimizer.zero_grad()
            pred_reg, pred_cls_list = model(curr_iq, tgt_iq)
            loss = 0.0
            
            if len(reg_indices) > 0:
                diff_reg = actual_diff[:, reg_indices]
                raw_l_reg = criterion_reg(pred_reg, diff_reg)
                loss += (raw_l_reg * reg_weights_tensor).mean()
                
            if len(cls_indices) > 0:
                diff_cls = actual_diff[:, cls_indices]
                l_cls_batch = 0.0
                for i, num_classes in enumerate(cls_num_classes_list):
                    target_val = diff_cls[:, i]
                    target_idx = torch.round((target_val + 1.0) / 2.0 * (num_classes - 1)).long()
                    target_idx = torch.clamp(target_idx, 0, num_classes - 1)
                    
                    raw_l_cls = criterion_cls(pred_cls_list[i], target_idx)
                    pred_idx = torch.argmax(pred_cls_list[i], dim=1)
                    distance_penalty = torch.abs(pred_idx - target_idx).float()
                    
                    weight = get_dynamic_weight(style_names[cls_indices[i]])
                    l_cls_batch += (raw_l_cls + 0.1 * distance_penalty).mean() * weight
                
                loss += (l_cls_batch / len(cls_num_classes_list))
            
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            
        scheduler.step()

        # --- [Validation Phase] ---
        # 10 에포크마다 혹은 마지막 에포크에 Validation Loss 계산
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss_total = 0.0
            
            with torch.no_grad():
                for v_curr_iq, v_tgt_iq, v_actual_diff in val_loader:
                    v_curr_iq, v_tgt_iq, v_actual_diff = v_curr_iq.to(device), v_tgt_iq.to(device), v_actual_diff.to(device)
                    
                    v_pred_reg, v_pred_cls_list = model(v_curr_iq, v_tgt_iq)
                    v_loss = 0.0
                    
                    if len(reg_indices) > 0:
                        v_diff_reg = v_actual_diff[:, reg_indices]
                        v_raw_l_reg = criterion_reg(v_pred_reg, v_diff_reg)
                        v_loss += (v_raw_l_reg * reg_weights_tensor).mean()
                        
                    if len(cls_indices) > 0:
                        v_diff_cls = v_actual_diff[:, cls_indices]
                        v_l_cls_batch = 0.0
                        for i, num_classes in enumerate(cls_num_classes_list):
                            v_target_val = v_diff_cls[:, i]
                            v_target_idx = torch.round((v_target_val + 1.0) / 2.0 * (num_classes - 1)).long()
                            v_target_idx = torch.clamp(v_target_idx, 0, num_classes - 1)
                            
                            v_raw_l_cls = criterion_cls(v_pred_cls_list[i], v_target_idx)
                            v_pred_idx = torch.argmax(v_pred_cls_list[i], dim=1)
                            v_distance_penalty = torch.abs(v_pred_idx - v_target_idx).float()
                            
                            v_weight = get_dynamic_weight(style_names[cls_indices[i]])
                            v_l_cls_batch += (v_raw_l_cls + 0.1 * v_distance_penalty).mean() * v_weight
                        
                        v_loss += (v_l_cls_batch / len(cls_num_classes_list))
                    
                    val_loss_total += v_loss.item()

            avg_train_loss = train_loss_total / len(train_loader)
            avg_val_loss = val_loss_total / len(val_loader)
            print(f"   Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
    print("[Success] Training procedure finished.\n")
    return model