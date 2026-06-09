import os
import pandas as pd
import numpy as np

datasets = ['emo', 'fake', 'hate', 'senti']
shots = ['zero_shot', 'few_shot']
models = ['qwen-72B', 'gemma2-27B']
results_dir = 'Results'

def clean_label(label):
    if pd.isna(label): return ""
    return str(label).strip().lower()

def fast_macro_f1(y_true, y_pred, unique_classes):
    f1s = []
    for c in unique_classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
            
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f1s.append(f1)
    return np.mean(f1s)

def bootstrap_ci(y_true, y_pred, unique_classes, n_iterations=1000, alpha=0.05):
    n = len(y_true)
    f1_scores = []
    np.random.seed(42)
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n, size=n)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        
        f1 = fast_macro_f1(sample_true, sample_pred, unique_classes)
        f1_scores.append(f1)
        
    f1_scores = np.sort(f1_scores)
    lower_bound = np.percentile(f1_scores, (alpha / 2) * 100)
    upper_bound = np.percentile(f1_scores, (1 - alpha / 2) * 100)
    
    return lower_bound, upper_bound

records = []

for ds in datasets:
    for shot in shots:
        file_path = os.path.join(results_dir, f"{ds}_{shot}.xlsx")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_excel(file_path)
        if 'label' not in df.columns:
            continue
            
        y_true = df['label'].apply(clean_label).values
        unique_classes = np.unique(y_true)
        
        for model in models:
            if model not in df.columns:
                continue
                
            y_pred = df[model].apply(clean_label).values
            base_f1 = fast_macro_f1(y_true, y_pred, unique_classes)
            lower, upper = bootstrap_ci(y_true, y_pred, unique_classes)
            
            records.append({
                'Dataset': ds,
                'Setting': shot,
                'Model': model,
                'Test_Size': len(y_true),
                'Macro_F1': base_f1,
                'CI_Lower': lower,
                'CI_Upper': upper
            })
            print(f"{ds} ({shot}) | {model} | F1: {base_f1:.4f} | 95% CI: [{lower:.4f}, {upper:.4f}]")

out_df = pd.DataFrame(records)
out_df.to_csv('Results/f1_confidence_intervals.csv', index=False)
print("Saved CI results to Results/f1_confidence_intervals.csv")
