import torch
import pandas as pd
import numpy as np

def evaluate_regressor(model, scaler_X, test_file, ref_file, iq_dim, style_names, out_file, restore, device):
    df = pd.read_csv(test_file)
    X_raw, y_norm = df.iloc[:, :iq_dim].values, df.iloc[:, iq_dim:].values
    X_scaled = scaler_X.transform(X_raw)
    n_samples = len(X_raw)

    ref_df = pd.read_csv(ref_file, index_col=0)
    ref_df.index = ref_df.index.astype(str).str.strip().str.upper()
    ref_col_map = {str(c).strip().upper(): c for c in ref_df.columns}
    ranges = np.array([float(ref_df.loc['MAX', ref_col_map[c.strip().upper()]]) - float(ref_df.loc['MIN', ref_col_map[c.strip().upper()]]) if c.strip().upper() in ref_col_map else 1.0 for c in style_names])

    # 확정적 Shift 짝짓기
    tgt_idx = (np.arange(n_samples) + 1) % n_samples
    actual_diff = y_norm[tgt_idx] - y_norm

    model.eval()
    with torch.no_grad():
        pred_diff = model(torch.tensor(X_scaled, dtype=torch.float32).to(device), torch.tensor(X_scaled[tgt_idx], dtype=torch.float32).to(device)).cpu().numpy()

    if restore == 1:
        pred_diff *= ranges
        actual_diff *= ranges

    err = np.abs(pred_diff - actual_diff)
    mae = np.mean(err, axis=0)

    rows = [["SUMMARY", "0_MAE"] + [None]*iq_dim + mae.tolist(), [None]*(2+iq_dim+len(style_names))]
    for i in range(n_samples):
        pid = f"S_{i:04d}_vs_{tgt_idx[i]:04d}"
        iq_v = X_raw[i].tolist()
        rows.extend([[pid, "1_Actual"] + iq_v + actual_diff[i].tolist(), [pid, "2_Pred"] + iq_v + pred_diff[i].tolist(), [pid, "3_Err"] + iq_v + err[i].tolist(), [None]*(2+iq_dim+len(style_names))])
    
    pd.DataFrame(rows, columns=["ID", "Type"] + list(df.columns[:iq_dim]) + list(style_names)).to_csv(out_file, index=False)\n