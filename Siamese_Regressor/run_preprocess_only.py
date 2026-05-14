import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from data_processing import prepare_raw_data

def select_file(title, filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    return filedialog.askopenfilename(title=title, filetypes=filetypes)

def main():
    print("="*50 + "\n [UI 기반 전처리 단독 실행 스크립트] \n" + "="*50)
    raw_file = select_file("1. 원본 데이터(Raw Data) 파일 선택")
    if not raw_file: return
    tgt_file = select_file("2. 타겟 컬럼 목록 파일 선택")
    if not tgt_file: return
    ref_file = select_file("3. Min/Max Reference 파일 선택")
    if not ref_file: return

    iq_dim = int(input("4. IQ Parameter 개수 (기본 80): ") or 80)
    test_size = float(input("5. Test 비율 (기본 0.1): ") or 0.1)
    train_out = input("6. Train 저장명 (기본 Train_Prep.csv): ") or "Train_Prep.csv"
    test_out = input("7. Test 저장명 (기본 Test_Prep.csv): ") or "Test_Prep.csv"
    
    target_columns = []
    df_t = pd.read_csv(tgt_file, header=None)
    target_columns = df_t.iloc[0, :].astype(str).tolist() if df_t.shape[1] > 1 and df_t.shape[0] == 1 else df_t.iloc[:, 0].astype(str).tolist()

    prepare_raw_data(raw_file, ref_file, target_columns, train_out, test_out, test_size, iq_dim)

if __name__ == "__main__": main()\n