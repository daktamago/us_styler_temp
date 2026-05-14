import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StyleDifferenceDataset

def run_training(model, X_tr, y_tr, X_va, y_va, batch_size=256, epochs=50, lr=1e-3, device='cuda'):
    print("\n[Training] Regressor 모델 학습 시작...")
    tr_loader = DataLoader(StyleDifferenceDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(StyleDifferenceDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for vc, vt, va in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(vc.to(device), vt.to(device)), va.to(device))
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()
        
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for vc, vt, va in va_loader:
                    va_loss += criterion(model(vc.to(device), vt.to(device)), va.to(device)).item()
            print(f"  Epoch [{ep+1}/{epochs}] Train Loss: {tr_loss/len(tr_loader):.4f} | Val Loss: {va_loss/len(va_loader):.4f}")
    return model\n