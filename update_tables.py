import re
import os

files_to_update = [
    r'e:\Evaluating the Effectiveness of Open-Source LLMs for Multi-Domain Bengali Text Classification with Zero-Shot and Few-Shot Prompting\Revised_Manuscript_with_Track_Changes___PLOS_ONE\plos_latex_template.tex',
    r'e:\Evaluating the Effectiveness of Open-Source LLMs for Multi-Domain Bengali Text Classification with Zero-Shot and Few-Shot Prompting\Review\Final_Manuscript.tex'
]

zero_f1_senti = ['49.62', '50.20', '49.90', '57.73', '58.66', '56.84']
zero_f1_emo = ['46.35', '45.18', '45.85', '57.90', '63.01', '66.80']
zero_f1_hate = ['54.22', '64.19', '56.40', '72.17', '76.11', '79.88']
zero_f1_fake = ['77.72', '55.45', '72.39', '76.91', '84.33', '82.91']

few_f1_senti = ['54.61', '48.09', '49.47', '56.44', '60.68', '58.82']
few_f1_emo = ['62.85', '63.07', '61.47', '70.56', '73.39', '78.25']
few_f1_hate = ['66.27', '58.93', '65.59', '81.48', '84.17', '83.68']
few_f1_fake = ['83.68', '82.02', '83.33', '84.45', '84.17', '83.68']

models = ['LLaMA-3.2-3B', 'Mistral-V3-7B', 'DeepSeek-R1-8B', 'Phi-4-14B', 'Gemma-2-27B', 'Qwen-2.5-72B']

def update_table(text, table_marker, f1_col1, f1_col2):

    idx = text.find(table_marker)
    if idx == -1: return text
    

    for i, model in enumerate(models):

        model_idx = text.find(model, idx)
        if model_idx == -1: continue
        end_idx = text.find(r'\\\hline', model_idx)
        line = text[model_idx:end_idx]
        

        parts = line.split('&')
        if len(parts) >= 9:

            parts[4] = f' {f1_col1[i]} '
            parts[8] = f' {f1_col2[i]} '
            new_line = '&'.join(parts)
            text = text[:model_idx] + new_line + text[end_idx:]
    return text

for filepath in files_to_update:
    if not os.path.exists(filepath): continue
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    

    text = update_table(text, r'\label{tab:zero_shot}', zero_f1_senti, zero_f1_emo)

    text = update_table(text, r'\multicolumn{4}{c|}{\textbf{Hate Speech Detection}}', zero_f1_hate, zero_f1_fake)
    

    text = update_table(text, r'\label{tab:few_shot}', few_f1_senti, few_f1_emo)

    text = update_table(text, r'\multicolumn{4}{c|}{\textbf{Hate Speech Detection}}', few_f1_hate, few_f1_fake)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Updated {filepath}")
