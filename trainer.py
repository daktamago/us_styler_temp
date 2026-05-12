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
        # ==========================================
        # [1] Training Phase
        # ==========================================
        model.train()
        train_loss_total = 0.0
        train_loss_reg = 0.0  # 복구됨: Regressor 손실 추적
        train_loss_cls = 0.0  # 복구됨: Classifier 손실 추적
        
        for curr_iq, tgt_iq, actual_diff in train_loader:
            curr_iq, tgt_iq, actual_diff = curr_iq.to(device), tgt_iq.to(device), actual_diff.to(device)
            optimizer.zero_grad()
            pred_reg, pred_cls_list = model(curr_iq, tgt_iq)
            loss = 0.0
            
            # Regression 파트
            if len(reg_indices) > 0:
                raw_l_reg = criterion_reg(pred_reg, actual_diff[:, reg_indices])
                l_reg = (raw_l_reg * reg_weights_tensor).mean()
                loss += l_reg
                train_loss_reg += l_reg.item()
                
            # Classification 파트
            if len(cls_indices) > 0:
                l_cls_batch = 0.0
                diff_cls = actual_diff[:, cls_indices]
                for i, num_classes in enumerate(cls_num_classes_list):
                    target_val = diff_cls[:, i]
                    t_idx = torch.clamp(torch.round((target_val + 1.0) / 2.0 * (num_classes - 1)).long(), 0, num_classes-1)
                    
                    raw_l_cls = criterion_cls(pred_cls_list[i], t_idx)
                    dist_penalty = torch.abs(torch.argmax(pred_cls_list[i], dim=1) - t_idx).float()
                    
                    weight = get_dynamic_weight(style_names[cls_indices[i]])
                    l_cls = (raw_l_cls + 0.1 * dist_penalty).mean() * weight
                    l_cls_batch += l_cls
                    
                l_cls_batch = l_cls_batch / len(cls_num_classes_list)
                loss += l_cls_batch
                train_loss_cls += l_cls_batch.item()
            
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            
        scheduler.step()

        # ==========================================
        # [2] Validation Phase
        # ==========================================
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss_total = 0.0
            val_loss_reg = 0.0  # 복구됨
            val_loss_cls = 0.0  # 복구됨
            
            with torch.no_grad():
                for v_curr, v_tgt, v_diff in val_loader:
                    v_curr, v_tgt, v_diff = v_curr.to(device), v_tgt.to(device), v_diff.to(device)
                    vp_reg, vp_cls_list = model(v_curr, v_tgt)
                    v_loss = 0.0
                    
                    if len(reg_indices) > 0:
                        v_l_reg = (criterion_reg(vp_reg, v_diff[:, reg_indices]) * reg_weights_tensor).mean()
                        v_loss += v_l_reg
                        val_loss_reg += v_l_reg.item()
                        
                    if len(cls_indices) > 0:
                        v_l_cls_batch = 0.0
                        vd_cls = v_diff[:, cls_indices]
                        for i, num_classes in enumerate(cls_num_classes_list):
                            vt_idx = torch.clamp(torch.round((vd_cls[:, i] + 1.0) / 2.0 * (num_classes - 1)).long(), 0, num_classes-1)
                            v_raw_cls = criterion_cls(vp_cls_list[i], vt_idx)
                            v_dist_pen = torch.abs(torch.argmax(vp_cls_list[i], dim=1) - vt_idx).float()
                            v_weight = get_dynamic_weight(style_names[cls_indices[i]])
                            v_l_cls_batch += (v_raw_cls + 0.1 * v_dist_pen).mean() * v_weight
                            
                        v_l_cls_batch = v_l_cls_batch / len(cls_num_classes_list)
                        v_loss += v_l_cls_batch
                        val_loss_cls += v_l_cls_batch.item()
                        
                    val_loss_total += v_loss.item()
            
            # ==========================================
            # [3] Logging (상세 출력 복구)
            # ==========================================
            t_len = len(train_loader)
            v_len = len(val_loader)
            
            avg_tr_tot = train_loss_total / t_len
            avg_tr_reg = train_loss_reg / t_len if len(reg_indices) > 0 else 0.0
            avg_tr_cls = train_loss_cls / t_len if len(cls_indices) > 0 else 0.0
            
            avg_va_tot = val_loss_total / v_len
            avg_va_reg = val_loss_reg / v_len if len(reg_indices) > 0 else 0.0
            avg_va_cls = val_loss_cls / v_len if len(cls_indices) > 0 else 0.0
            
            print(f"   Epoch [{epoch+1}/{epochs}]")
            print(f"      [Train] Tot: {avg_tr_tot:.4f} | Reg: {avg_tr_reg:.4f} | Cls: {avg_tr_cls:.4f}")
            print(f"      [Val]   Tot: {avg_va_tot:.4f} | Reg: {avg_va_reg:.4f} | Cls: {avg_va_cls:.4f}")
            
    print("[Success] Training procedure finished.\n")
    return model