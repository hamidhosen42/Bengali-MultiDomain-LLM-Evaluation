import pandas as pd
import os

file_path = os.path.join('Results', 'emo_few_shot.xlsx')
if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    if 'label' in df.columns:
        print("Emotion Dataset Distribution:")
        print(df['label'].value_counts())
    else:
        print("Label column not found")
else:
    print("File not found")
