import os
import pandas as pd
import numpy as np

datasets = ['emo', 'fake', 'hate', 'senti']
shots = ['zero_shot', 'few_shot']
model1 = 'qwen-72B'
model2 = 'gemma2-27B'

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
        
        # To avoid division by zero
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

def fast_permutation_test(y_true, y_pred1, y_pred2, unique_classes, n_iterations=1000):
    n = len(y_true)
    
    # Base Accuracy
    acc_m1 = np.mean(y_true == y_pred1)
    acc_m2 = np.mean(y_true == y_pred2)
    acc_diff = abs(acc_m1 - acc_m2)
    
    # Base F1
    f1_m1 = fast_macro_f1(y_true, y_pred1, unique_classes)
    f1_m2 = fast_macro_f1(y_true, y_pred2, unique_classes)
    f1_diff = abs(f1_m1 - f1_m2)
    
    acc_count = 0
    f1_count = 0
    
    # Pre-generate swaps
    np.random.seed(42)
    swaps = np.random.randint(0, 2, size=(n_iterations, n)).astype(bool)
    
    for i in range(n_iterations):
        swap = swaps[i]
        sim_pred1 = np.where(swap, y_pred2, y_pred1)
        sim_pred2 = np.where(swap, y_pred1, y_pred2)
        
        # Acc
        sim_acc_diff = abs(np.mean(y_true == sim_pred1) - np.mean(y_true == sim_pred2))
        if sim_acc_diff >= acc_diff:
            acc_count += 1
            
        # F1
        sim_f1_m1 = fast_macro_f1(y_true, sim_pred1, unique_classes)
        sim_f1_m2 = fast_macro_f1(y_true, sim_pred2, unique_classes)
        if abs(sim_f1_m1 - sim_f1_m2) >= f1_diff:
            f1_count += 1
            
    return (acc_m1, acc_m2, acc_count / n_iterations,
            f1_m1, f1_m2, f1_count / n_iterations)

records = []

for ds in datasets:
    for shot in shots:
        file_path = os.path.join(results_dir, f"{ds}_{shot}.xlsx")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_excel(file_path)
        if 'label' not in df.columns or model1 not in df.columns or model2 not in df.columns:
            continue
            
        y_true = df['label'].apply(clean_label).values
        y_pred1 = df[model1].apply(clean_label).values
        y_pred2 = df[model2].apply(clean_label).values
        
        unique_classes = np.unique(y_true)
        
        acc_1, acc_2, p_acc, f1_1, f1_2, p_f1 = fast_permutation_test(y_true, y_pred1, y_pred2, unique_classes)
        
        records.append({
            'Dataset': ds,
            'Setting': shot,
            'Model1': model1,
            'Model2': model2,
            'Acc_M1': acc_1,
            'Acc_M2': acc_2,
            'Acc_p_val': p_acc,
            'F1_M1': f1_1,
            'F1_M2': f1_2,
            'F1_p_val': p_f1
        })
        print(f"Done: {ds} {shot} - Acc p={p_acc:.4f}, F1 p={p_f1:.4f}")

out_df = pd.DataFrame(records)
out_df.to_excel('significance_tests.xlsx', index=False)
