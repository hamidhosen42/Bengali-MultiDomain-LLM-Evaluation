import pandas as pd

df_test = pd.read_excel('Datasets/emo_test.xlsx')
print("emo_test columns:", df_test.columns.tolist())

df_res = pd.read_excel('Results/emo_few_shot.xlsx')
print("emo_few_shot columns:", df_res.columns.tolist())

df_res_qwen_gemma = pd.read_excel('Results/senti_zero_shot.xlsx')
print("senti_zero_shot columns:", df_res_qwen_gemma.columns.tolist())

