import torch, joblib, pandas as pd
from model import SiameseRegressor
from evaluator import evaluate_regressor

def main():
    print("="*40 + "\n [Regressor Only] 단독 평가 스크립트 \n" + "="*40)
    model_path = input("1. 모델(.pth) 경로: ") or "model_Regressor.pth"
    scaler_path = input("2. 스케일러(.pkl) 경로: ") or "scaler_reg.pkl"
    te_file = input("3. Test 데이터 경로: ") or "IQ_Style_Test.csv"
    ref_file = input("4. Ref 파일 경로: ") or "ParameterMinMaxStep.csv"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chkpt = torch.load(model_path, map_location=device)
    conf = chkpt['config']
    
    model = SiameseRegressor(conf['input_dim'], conf['hidden_dims'], conf['extractor_dims'], conf['reg_head_dims'], conf['reg_dim']).to(device)
    model.load_state_dict(chkpt['model_state_dict'])
    
    style_cols = pd.read_csv(te_file).columns[conf['input_dim']:]
    evaluate_regressor(model, joblib.load(scaler_path), te_file, ref_file, conf['input_dim'], style_cols, "eval_only_results.csv", 1, device)
    print("✅ 단독 평가 완료!")

if __name__ == "__main__": main()\n