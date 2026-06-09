import os
import pandas as pd

datasets = ['emo', 'fake', 'hate', 'senti']
results_dir = 'Results'
models = ['llama32-3B', 'mistral-7B', 'phi4-14B', 'qwen-72B', 'gemma2-27B', 'deepseek-8B']

def clean_label(label):
    if pd.isna(label): return ""
    return str(label).strip().lower()

for ds in datasets:
    # Using few-shot setting for error analysis
    file_path = os.path.join(results_dir, f"{ds}_few_shot.xlsx")
    if not os.path.exists(file_path):
        continue
        
    df = pd.read_excel(file_path)
    if 'label' not in df.columns or 'sentence' not in df.columns:
        continue
        
    df['true_label'] = df['label'].apply(clean_label)
    
    # We will extract misclassifications for ALL models
    for model in models:
        if model in df.columns:
            df[f'{model}_pred'] = df[model].apply(clean_label)
            
            # Filter where prediction does not match true label
            misclassified = df[df['true_label'] != df[f'{model}_pred']].copy()
            
            # Save to file
            out_file = os.path.join(results_dir, f"error_analysis_{ds}_{model}.xlsx")
            misclassified[['sentence', 'true_label', f'{model}_pred']].to_excel(out_file, index=False)
            print(f"Saved {len(misclassified)} misclassifications for {ds} ({model}) to {out_file}")
