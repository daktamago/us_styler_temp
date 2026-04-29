import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StyleDifferenceDataset  # 분리된 데이터셋 모듈 Import

def run_multitask_training_pipeline(model, X_train, y_train, X_val, y_val, 
                                    reg_indices, cls_indices, cls_num_classes_list,
                                    batch_size=256, epochs=50, lr=1e-3, device='cuda'):
    
    print(f"\n🚀 [Multi-Task Hybrid 모델] 학습을 시작합니다...")
    
    train_loader = DataLoader(StyleDifferenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(StyleDifferenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    # 1. Loss Functions (Multi-Task)
    criterion_reg = nn.HuberLoss() # 또는 nn.L1Loss()
    criterion_cls = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss 가중치 (필요 시 조절)
    alpha_reg = 1.0
    beta_cls = 1.0 
    
    for epoch in range(epochs):
        model.train()
        train_loss_total, train_loss_reg, train_loss_cls = 0.0, 0.0, 0.0
        
        for curr_iq, tgt_iq, actual_diff in train_loader:
            curr_iq, tgt_iq, actual_diff = curr_iq.to(device), tgt_iq.to(device), actual_diff.to(device)
            
            # 정답 데이터를 Reg 그룹과 Cls 그룹으로 분할
            diff_reg = actual_diff[:, reg_indices] if len(reg_indices) > 0 else None
            diff_cls = actual_diff[:, cls_indices] if len(cls_indices) > 0 else None
            
            optimizer.zero_grad()
            
            # 모델 추론
            pred_reg, pred_cls_list = model(curr_iq, tgt_iq)
            
            loss = 0.0
            
            # A. Regression Loss 계산
            if diff_reg is not None and diff_reg.numel() > 0:
                l_reg = criterion_reg(pred_reg, diff_reg)
                loss += alpha_reg * l_reg
                train_loss_reg += l_reg.item()
                
            # B. Ordinal Classification Loss 계산
            if diff_cls is not None and diff_cls.numel() > 0:
                l_cls_total = 0.0
                # 각 Classification 파라미터별로 Loss 계산
                for i, num_classes in enumerate(cls_num_classes_list):
                    # -1.0 ~ 1.0 사이의 연속형 차이값을 0 ~ (num_classes-1) 사이의 정수 인덱스로 변환
                    # 공식: round((val + 1) / 2 * (K - 1))
                    target_val = diff_cls[:, i]
                    target_idx = torch.round((target_val + 1.0) / 2.0 * (num_classes - 1)).long()
                    
                    # 범위를 벗어나지 않도록 안전장치 (Clamp)
                    target_idx = torch.clamp(target_idx, 0, num_classes - 1)
                    
                    # Cross Entropy 계산 (Logit vs Integer Index)
                    l_cls = criterion_cls(pred_cls_list[i], target_idx)
                    
                    # [심화: Ordinal Penalty 부여 (선택)]
                    # 정답 클래스와 예측 클래스의 거리가 멀수록 페널티를 주는 장치
                    pred_idx = torch.argmax(pred_cls_list[i], dim=1)
                    distance_penalty = torch.abs(pred_idx - target_idx).float().mean()
                    
                    # 기본 CE Loss에 거리 페널티를 더해 서수적(Ordinal) 특성 강화
                    l_cls_total += (l_cls + 0.1 * distance_penalty)
                
                # 파라미터 개수로 나누어 평균 Loss 산출
                l_cls_total = l_cls_total / len(cls_num_classes_list)
                loss += beta_cls * l_cls_total
                train_loss_cls += l_cls_total.item()
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            
        scheduler.step()
        
        # 로그 출력 (10 Epoch마다)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            avg_tot = train_loss_total / len(train_loader)
            avg_reg = train_loss_reg / len(train_loader) if len(reg_indices) > 0 else 0
            avg_cls = train_loss_cls / len(train_loader) if len(cls_indices) > 0 else 0
            print(f"   Epoch [{epoch+1}/{epochs}] Total Loss: {avg_tot:.4f} | Reg Loss: {avg_reg:.4f} | Cls Loss: {avg_cls:.4f}")
            
    print("✅ 학습 완료!\n")
    return model