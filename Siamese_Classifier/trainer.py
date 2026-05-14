import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StyleDifferenceDataset

def run_training(model, X_tr, y_tr, X_va, y_va, cls_num_list, batch_size=256, epochs=50, lr=1e-3, device='cuda'):
    print("\n[Training] Classifier 모델 학습 시작...")
    tr_loader = DataLoader(StyleDifferenceDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(StyleDifferenceDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for vc, vt, va in tr_loader:
            vc, vt, va = vc.to(device), vt.to(device), va.to(device)
            optimizer.zero_grad()
            preds = model(vc, vt)
            loss = 0.0
            for i, num_cls in enumerate(cls_num_list):
                t_idx = torch.clamp(torch.round((va[:, i] + 1.0) / 2.0 * (num_cls - 1)).long(), 0, num_cls-1)
                loss += criterion(preds[i], t_idx) + 0.1 * torch.abs(torch.argmax(preds[i], dim=1) - t_idx).float().mean()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()
        
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for vc, vt, va in va_loader:
                    preds = model(vc.to(device), vt.to(device))
                    for i, num_cls in enumerate(cls_num_list):
                        t_idx = torch.clamp(torch.round((va.to(device)[:, i] + 1.0) / 2.0 * (num_cls - 1)).long(), 0, num_cls-1)
                        va_loss += (criterion(preds[i], t_idx) + 0.1 * torch.abs(torch.argmax(preds[i], dim=1) - t_idx).float().mean()).item()
            print(f"  Epoch [{ep+1}/{epochs}] Train Loss: {tr_loss/len(tr_loader):.4f} | Val Loss: {va_loss/len(va_loader):.4f}")
    return model\n