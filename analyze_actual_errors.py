import os
import pandas as pd
import random

results_dir = 'Results'
datasets = ['emo', 'fake', 'hate', 'senti']
model = 'qwen-72B'

with open('error_samples.txt', 'w', encoding='utf-8') as f:
    for ds in datasets:
        file_path = os.path.join(results_dir, f"error_analysis_{ds}_{model}.xlsx")
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            f.write(f"\n=== {ds.upper()} DATASET (Misclassified by {model}) ===\n")
            # Get up to 15 random samples to find good patterns
            samples = df.sample(min(15, len(df)), random_state=42)
            for _, row in samples.iterrows():
                f.write(f"TEXT: {row['sentence']}\n")
                f.write(f"TRUE: {row['true_label']} | PRED: {row[f'{model}_pred']}\n")
                f.write("-" * 50 + "\n")
