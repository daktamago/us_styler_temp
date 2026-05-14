import torch, joblib
from data_processing import load_scale_and_group_data, get_class_numbers
from model import SiameseClassifier
from trainer import run_training
from evaluator import evaluate_classifier

def main():
    print("="*40 + "\n [Classifier Only] 파이프라인 \n" + "="*40)
    tr_file = input("1. Train 파일 (기본 IQ_Style_Train.csv): ") or "IQ_Style_Train.csv"
    te_file = input("2. Test 파일 (기본 IQ_Style_Test.csv): ") or "IQ_Style_Test.csv"
    ref_file = input("3. Ref 파일 (기본 ParameterMinMaxStep.csv): ") or "ParameterMinMaxStep.csv"
    iq_dim = int(input("4. IQ 개수 (기본 80): ") or 80)
    hid_dims = [int(x) for x in (input("5. Encoder 차원 (기본 512,1024,1024): ") or "512,1024,1024").split(',')]
    ext_dims = [int(x) for x in (input("6. Extractor 차원 (기본 256,256,128): ") or "256,256,128").split(',')]
    cls_dims = [int(x) for x in (input("7. Classifier Head 차원 (기본 128): ") or "128").split(',')]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr, X_va, y_tr, y_va, scaler_X, style_cols = load_scale_and_group_data(tr_file, iq_dim)
    joblib.dump(scaler_X, "scaler_cls.pkl")

    cls_num_list = get_class_numbers(ref_file, style_cols)
    model = SiameseClassifier(iq_dim, hid_dims, ext_dims, cls_dims, cls_num_list).to(device)
    model = run_training(model, X_tr, y_tr, X_va, y_va, cls_num_list, device=device)
    
    chkpt = {'model_state_dict': model.state_dict(), 'config': {'input_dim': iq_dim, 'hidden_dims': hid_dims, 'extractor_dims': ext_dims, 'cls_head_dims': cls_dims, 'cls_num_list': cls_num_list}}
    torch.save(chkpt, 'model_Classifier.pth')
    
    evaluate_classifier(model, scaler_X, te_file, ref_file, cls_num_list, iq_dim, style_cols, "eval_classifier.csv", 1, device)
    print("✅ 모든 프로세스 완료!")

if __name__ == "__main__": main()\n
