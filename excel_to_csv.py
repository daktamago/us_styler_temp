import pandas as pd

# 1. 기존 엑셀 로드
df = pd.read_excel('IQ_Style_Data.xlsx')

# 2. 빠른 포맷으로 저장 (둘 중 하나 선택)
df.to_csv('training_data.csv', index=False)
# df.to_parquet('training_data.parquet', index=False) # pyarrow 설치 필요