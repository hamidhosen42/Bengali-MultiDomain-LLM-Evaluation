from transformers import set_seed
import torch, transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import json,re, ast,os,time
from sklearn.metrics import classification_report, confusion_matrix

set_seed(42)



current_dir = os.getcwd()

root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  
root_dir = os.path.abspath(os.path.join(root_dir,''))  
dataset_dir = os.path.join(root_dir,'Datasets')
results_dir = os.path.join(root_dir,'Results')
prompt_dir = os.path.join(root_dir,'Prompts')


llm_names = ['llama32-3B', 'mistral-7B','deepseek-8B',  'phi4-14B','gemma2-27B', 'qwen-72B']
# display names used in all figures (identical to the manuscript)
display_names = {'llama32-3B': 'LLaMA-3.2-3B', 'mistral-7B': 'Mistral-V3-7B', 'deepseek-8B': 'DeepSeek-R1-8B', 'phi4-14B': 'Phi-4-14B', 'gemma2-27B': 'Gemma-2-27B', 'qwen-72B': 'Qwen-2.5-72B'}

def classwise_f1_pivot(data, classes):

    f1_list, category, llms, mf1_list = [], [], [], []
    for col in llm_names:
        d = classification_report(y_true=data['label'], y_pred=data[col],
                                digits=4, output_dict=True)
        
        mf1 = "{:.2f}".format(d['macro avg']['f1-score'] * 100)
        
        for label in classes:
            f1 = "{:.2f}".format(d[label]['f1-score'] * 100)
            f1_list.append(float(f1))
            category.append(label.capitalize())
            llms.append(col.capitalize())
            
        mf1_list.append((col, mf1))    

    # making pivot
    make_pivot = pd.DataFrame({'F1-Score': f1_list, 'Category': category, 'LLMs': llms})
    df_pivot = make_pivot.pivot(index='LLMs', columns='Category', values='F1-Score').reset_index()
    df = pd.melt(df_pivot, id_vars='LLMs', var_name="Category", value_name="Values")

    # LLM names
    u_llm = ['Llama32-3b', 'Mistral-7b', 'Deepseek-8b', 'Phi4-14b', 'Gemma2-27b', 'Qwen-72b']
    
    # Ensure LLMs is a Categorical type with your desired order
    df['LLMs'] = pd.Categorical(df['LLMs'], categories=u_llm, ordered=True)

    # Sort by this custom category
    df = df.sort_values('LLMs') 

    return df, mf1_list


# Zero Shot Sentiment Results
zero_senti = pd.read_excel(os.path.join(results_dir,'senti_zero_shot.xlsx'))
# Few Shot Sentiment Results
few_senti = pd.read_excel(os.path.join(results_dir,'senti_few_shot.xlsx'))

# Class Names
senti_classes = ['positive', 'negative', 'neutral'] 
# zero_senti

## Classification Report

for col in llm_names:
    print('-----------')
    print(col)
    print('------------')
    print(classification_report(y_true=zero_senti['label'],y_pred=zero_senti[col],digits=4))
    print(classification_report(y_true=few_senti['label'],y_pred=few_senti[col],digits=4))

import matplotlib.pyplot as plt
import seaborn as sns
import os

def con_mat_senti(cm, class_names, model_name, results_dir="."):
    """
    Draws and saves a clean confusion matrix heatmap with dark blue color tones.
    """

    plt.figure(figsize=(4, 3))  # 🔹 Bigger and clearer
    ax = plt.subplot()

    # 🔹 Use dark blue color map ('Blues' similar to your provided image)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        annot_kws={"size": 12, "weight": "bold", "color": "white"},
        square=True,
        cbar=True,
        linewidths=0.4,
        linecolor='white'
    )

    # 🔹 Label settings
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=6)
    ax.set_ylabel("True label", fontsize=11, labelpad=6)

    ax.xaxis.set_ticklabels(class_names, rotation=0, fontsize=10, weight='bold')
    ax.yaxis.set_ticklabels(class_names, rotation=0, fontsize=10, weight='bold')

    # 🔹 Adjust layout
    plt.tight_layout()

    # 🔹 Save high-quality figure
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"{model_name}.png"), dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()

## Zero shot worst model 
v_model = confusion_matrix(zero_senti['label'],zero_senti['llama32-3B'])
class_names =['Negative','Neutral','Positive']
con_mat_senti(v_model, class_names,"1_senti_llama_conf_zero_worst_model")
#v_model

## Zero shot best model 
v_model = confusion_matrix(zero_senti['label'],zero_senti['gemma2-27B'])
class_names =['Negative','Neutral','Positive']
con_mat_senti(v_model, class_names,"1_senti_gemma_conf_zero_best_model")
#v_model

## Few shot worst model 
v_model = confusion_matrix(few_senti['label'],few_senti['mistral-7B'])
class_names =['Negative','Neutral','Positive']
con_mat_senti(v_model, class_names,"1_senti_mistral_conf_few_worst_model")
#v_model

## Few shot best model 
v_model = confusion_matrix(few_senti['label'],few_senti['gemma2-27B'])
class_names =['Negative','Neutral','Positive']
con_mat_senti(v_model, class_names,"1_senti_gemma_conf_few_best_model")
#v_model

## 🎨 Clean Classwise Performance Plot (Solid Color Style)
def classwise_plot(df, plt_name, y_limit, figsize, save=False):
    model_names = [
        'LLaMA-3.2-3B', 'Mistral-V3-7B', 'DeepSeek-R1-8B',
        'Phi-4-14B', 'Gemma-2-27B', 'Qwen-2.5-72B'
    ]

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Categories and solid colors
    categories = df['Category'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    # Bar setup
    bar_width = 0.25
    x = np.arange(len(model_names))

    # Loop through each category to plot bars
    for i, cat in enumerate(categories):
        values = df[df['Category'] == cat]['Values'].values
        bars = ax.bar(x + i * bar_width, values, width=bar_width,
                      label=cat, color=colors[i], edgecolor='black', linewidth=0.8)

        # 🧾 Value labels (clear and bold)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f'{height:.1f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='black')

    # 🎯 Axis and labels
    ax.set_xlabel('')
    ax.set_ylabel('F1-Score (%)', fontsize=10)
    # ax.set_title('Scores by Model and Category', fontsize=11, pad=10)
    ax.set_ylim(y_limit)

    # X-axis model names
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)

    # Legend styling (clean)
    ax.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=3, fontsize=9, frameon=True)

    # Grid and borders
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save figure if requested
    if save:
        plt.savefig(os.path.join(results_dir, f'{plt_name}.png'),
                    dpi=500, bbox_inches='tight')

    plt.show()


## zero shot pivot
df_sz,llm_sz = classwise_f1_pivot(zero_senti,senti_classes)
## few shot pivot
df_sf,llm_sf = classwise_f1_pivot(few_senti,senti_classes)

llm_sz

llm_sf

sz = [('llama32-3B', '51.83'),('mistral-7B', '52.25'), ('deepseek-8B', '52.15'), ('phi4-14B', '60.28'), ('gemma2-27B', '61.54'), ('qwen-72B', '59.32')]
sf = [('llama32-3B', '57.19'),('mistral-7B', '49.71'), ('deepseek-8B', '51.71'), ('phi4-14B', '58.88'), ('gemma2-27B', '63.00'), ('qwen-72B', '61.30')]

# Classwise performace for zero shot
classwise_plot(df_sz,'1_senti_classwise_zero',y_limit=[30,90],figsize=(4, 2.5),save = True)


# Classwise performace for zero shot
classwise_plot(df_sf,'1_senti_classwise_few',y_limit=[30,90],figsize=(4, 2.5), save = True)

# Zero Shot Emotion Results
zero_emo = pd.read_excel(os.path.join(results_dir,'emo_zero_shot.xlsx'))
few_emo = pd.read_excel(os.path.join(results_dir,'emo_few_shot.xlsx'))
emo_classes = ['disgust', 'sadness', 'joy', 'fear', 'surprise', 'anger']
zero_emo.head(5)

## Classification Report

for col in llm_names:
    print('-----------')
    print(col)
    print('------------')
    print(classification_report(y_true=zero_emo['label'],y_pred=zero_emo[col],digits=4))
    print(classification_report(y_true=few_emo['label'],y_pred=few_emo[col],digits=4))
    # print('--------------------')

## zero shot pivot
df_ez,llm_ez = classwise_f1_pivot(zero_emo,emo_classes)
## few shot pivot
df_ef,llm_ef = classwise_f1_pivot(few_emo,emo_classes)

llm_ef

ez = [('llama32-3B', '44.28'),('mistral-7B', '33.89'), ('deepseek-8B', '42.14'),('phi4-14B', '56.96'),('gemma2-27B', '58.40'), ('qwen-72B', '59.44')]
ef = [('llama32-3B', '47.12'), ('mistral-7B', '46.41'), ('deepseek-8B', '43.41'),('phi4-14B', '54.56'),('gemma2-27B', '60.27'),('qwen-72B', '65.86')]

## 🎯 Emotion Recognition Classwise Performance Plot (Clean & Publication-Ready)
def classwise_plot_emo(df, plt_name, y_limit, save=False):

    # ✅ Model names
    model_names = [
        'LLaMA-3.2-3B', 'Mistral-V3-7B', 'DeepSeek-R1-8B',
        'Phi-4-14B', 'Gemma-2-27B', 'Qwen-2.5-72B'
    ]

    plt.figure(figsize=(8, 3))
    ax = plt.gca()

    # 🎨 Define solid color palette for clarity
    palette_colors = ['#4C72B0', '#DD8452', '#55A868']  # Blue, Orange, Green

    # 🧱 Draw bar plot
    splot = sns.barplot(
        data=df,
        x='LLMs', y='Values', hue='Category',
        palette=palette_colors, edgecolor='black', linewidth=0.7
    )

    # 🧩 Annotate bars with vertical values
    for bar in splot.patches:
        height = bar.get_height()
        if not np.isnan(height):
            x = bar.get_x() + bar.get_width() / 2
            splot.annotate(f'{height:.1f}',
                          xy=(x, height),
                          xytext=(0, 5),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8, rotation=90, fontweight='bold')

    # 🪶 Axis labels and limits
    ax.set_xlabel('')
    ax.set_ylabel('F1-Score (%)', fontsize=10)
    ax.set_ylim(y_limit)


    # 🧭 Customize axes and style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("gray")
    ax.spines['bottom'].set_color("gray")
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    # 📊 X-axis labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)

    # 🧭 Legend style
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        fontsize=9,
        frameon=True
    )

    plt.tight_layout()

    # 💾 Save the figure if needed
    if save:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f'{plt_name}.png'),
                    dpi=500, bbox_inches='tight')

    plt.show()

# classwise performance
classwise_plot_emo(df_ez,'2_emo_classwise_zero', y_limit=[5,130], save = True)

# classwise performance
classwise_plot_emo(df_ef,'2_emo_classwise_few', y_limit=[5,130], save = True)

## Confusion matrix function
def con_mat_emo(cm, class_names,model_name):
    plt.figure(figsize=(6, 4.5))  # 🔹 Bigger and clearer
    ax = plt.subplot()

    # 🔹 Use dark blue color map ('Blues' similar to your provided image)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        annot_kws={"size": 12, "weight": "bold", "color": "white"},
        square=True,
        cbar=True,
        linewidths=0.4,
        linecolor='white'
    )

    # 🔹 Label settings
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=6)
    ax.set_ylabel("True label", fontsize=11, labelpad=6)

    ax.xaxis.set_ticklabels(class_names, rotation=0, fontsize=10, weight='bold')
    ax.yaxis.set_ticklabels(class_names, rotation=0, fontsize=10, weight='bold')

    # 🔹 Adjust layout
    plt.tight_layout()

    # 🔹 Save high-quality figure
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"{model_name}.png"), dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()

## zero shot worst model
v_model = confusion_matrix(zero_emo['label'],zero_emo['mistral-7B'])
class_names =['Anger','Disgust','Fear','Joy','Sadness','Surprise']
con_mat_emo(v_model, class_names,"2_emo_mistral_conf_zero_worst_model")
#v_model

## zero shot best model
v_model = confusion_matrix(zero_emo['label'],zero_emo['qwen-72B'])
class_names =['Anger','Disgust','Fear','Joy','Sadness','Surprise']
con_mat_emo(v_model, class_names,"2_emo_qwen_conf_zero_best_model")
#v_model

## few shot worst model
v_model = confusion_matrix(few_emo['label'],few_emo['deepseek-8B'])
class_names =['Anger','Disgust','Fear','Joy','Sadness','Surprise']
con_mat_emo(v_model, class_names,"2_emo_deep_conf_few_worst_model")
#v_model

## few shot best model
v_model = confusion_matrix(few_emo['label'],few_emo['qwen-72B'])
class_names =['Anger','Disgust','Fear','Joy','Sadness','Surprise']
con_mat_emo(v_model, class_names,"2_emo_qwen_conf_few_best_model")
#v_model

# Zero/Few Shot hate speech Results
zero_hate = pd.read_excel(os.path.join(results_dir,'hate_zero_shot.xlsx'))
few_hate = pd.read_excel(os.path.join(results_dir,'hate_few_shot.xlsx'))
hate_classes = ['hate', 'not-hate']
zero_hate.head(5)

## classification report
for col in llm_names:
    print('-----------')
    print(col)
    print('------------')
    print(classification_report(y_true=zero_hate['label'],y_pred=zero_hate[col],digits=4))
    print(classification_report(y_true=few_hate['label'],y_pred=few_hate[col],digits=4))
    # print('--------------------')

## zero shot pivot
df_hz,llm_hz = classwise_f1_pivot(zero_hate,hate_classes)
## few shot pivot
df_hf,llm_hf = classwise_f1_pivot(few_hate,hate_classes)

llm_hf

llm_hz

hz = [('llama32-3B', '54.22'),('mistral-7B', '64.19'), ('deepseek-8B', '56.40'), ('phi4-14B', '72.17'),('gemma2-27B', '76.11'),('qwen-72B', '79.88')]
hf = [('llama32-3B', '62.85'),('mistral-7B', '63.07'), ('deepseek-8B', '61.47'),('phi4-14B', '70.56'),('gemma2-27B', '73.39'),('qwen-72B', '78.25')]

## Class wise Performance Plot
def classwise_plot_hate(df, plt_name, y_limit ,save = False):

    model_names = ['LLaMA-3.2-3B','Mistral-V3-7B','DeepSeek-R1-8B','Phi-4-14B','Gemma-2-27B','Qwen-2.5-72B']
    plt.figure(figsize=(8, 3))
    ax = plt.gca()

    # 🎨 Define solid color palette for clarity
    palette_colors = ['#4C72B0', '#DD8452', '#55A868']  # Blue, Orange, Green

    # 🧱 Draw bar plot
    splot = sns.barplot(
        data=df,
        x='LLMs', y='Values', hue='Category',
        palette=palette_colors, edgecolor='black', linewidth=0.7
    )

    # 🧩 Annotate bars with vertical values
    for bar in splot.patches:
        height = bar.get_height()
        if not np.isnan(height):
            x = bar.get_x() + bar.get_width() / 2
            splot.annotate(f'{height:.1f}',
                          xy=(x, height),
                          xytext=(0, 5),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8, rotation=90, fontweight='bold')

    # 🪶 Axis labels and limits
    ax.set_xlabel('')
    ax.set_ylabel('F1-Score (%)', fontsize=10)
    ax.set_ylim(y_limit)


    # 🧭 Customize axes and style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("gray")
    ax.spines['bottom'].set_color("gray")
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    # 📊 X-axis labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)

    # 🧭 Legend style
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        fontsize=9,
        frameon=True
    )

    plt.tight_layout()

    # 💾 Save the figure if needed
    if save:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f'{plt_name}.png'),
                    dpi=500, bbox_inches='tight')

    plt.show()

# classwise performance
classwise_plot_hate(df_hz,'3_hate_classwise_zero',y_limit=[20,90],save = True)

# classwise performance
classwise_plot_hate(df_hf,'3_hate_classwise_few',y_limit=[20,90],save = True)

## Confusion matrix function
def con_mat_hate(cm, class_names,model_name):
    plt.figure(figsize=(4, 3))  # 🔹 Bigger and clearer
    ax = plt.subplot()

    # 🔹 Use dark blue color map ('Blues' similar to your provided image)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        annot_kws={"size": 12, "weight": "bold", "color": "white"},
        square=True,
        cbar=True,
        linewidths=0.4,
        linecolor='white'
    )

    # 🔹 Label settings
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=6)
    ax.set_ylabel("True label", fontsize=11, labelpad=6)

    ax.xaxis.set_ticklabels(class_names, rotation=0, fontsize=10, weight='bold')
    ax.yaxis.set_ticklabels(class_names, rotation=0, fontsize=10, weight='bold')

    # 🔹 Adjust layout
    plt.tight_layout()

    # 🔹 Save high-quality figure
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"{model_name}.png"), dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()


## zero shot worst model
v_model = confusion_matrix(zero_hate['label'],zero_hate['llama32-3B'])
class_names =['Hate','Not-Hate']
con_mat_hate(v_model, class_names,"3_hate_llama_conf_zero_worst_model")
#v_model

## zero shot best model
v_model = confusion_matrix(zero_hate['label'],zero_hate['qwen-72B'])
class_names =['Hate','Not-Hate']
con_mat_hate(v_model, class_names,"3_hate_qwen_conf_zero_best_model")
#v_model

## few shot worst model
v_model = confusion_matrix(few_hate['label'],few_hate['deepseek-8B'])
class_names =['Hate','Not-Hate']
con_mat_hate(v_model, class_names,"3_hate_deep_conf_few_worst_model")
#v_model

## few shot best model
v_model = confusion_matrix(few_hate['label'],few_hate['qwen-72B'])
class_names =['Hate','Not-Hate']
con_mat_hate(v_model, class_names,"3_hate_qwen_conf_few_best_model")
#v_model

# Zero/Few Shot hate speech Results
zero_fake = pd.read_excel(os.path.join(results_dir,'fake_zero_shot.xlsx'))
few_fake = pd.read_excel(os.path.join(results_dir,'fake_few_shot.xlsx'))
fake_classes = ['fake', 'real']
zero_fake.head(5)

## classification report
for col in llm_names:
    print('-----------')
    print(col)
    print('------------')
    print(classification_report(y_true=zero_fake['label'],y_pred=zero_fake[col],digits=4))
    print(classification_report(y_true=few_fake['label'],y_pred=few_fake[col],digits=4))
    # print('--------------------')

## zero shot pivot
df_fz,llm_fz = classwise_f1_pivot(zero_fake,fake_classes)
## few shot pivot
df_ff,llm_ff = classwise_f1_pivot(few_fake,fake_classes)

llm_ff

llm_fz

fz = [('llama32-3B', '77.72'),('mistral-7B', '55.45'),('deepseek-8B', '72.39'),('phi4-14B', '76.91'),('gemma2-27B', '84.33'), ('qwen-72B', '82.91')]
ff = [('llama32-3B', '66.27'),('mistral-7B', '58.93'), ('deepseek-8B', '65.59'), ('phi4-14B', '81.48'), ('gemma2-27B', '84.17'),('qwen-72B', '83.68')]

## Class wise Performance Plot
def classwise_plot_fake(df, plt_name, y_limit ,save = False):

    model_names = ['LLaMA-3.2-3B','Mistral-V3-7B','DeepSeek-R1-8B','Phi-4-14B','Gemma-2-27B','Qwen-2.5-72B']
    plt.figure(figsize=(8, 3))
    ax = plt.gca()

    # 🎨 Define solid color palette for clarity
    palette_colors = ['#4C72B0', '#DD8452', '#55A868']  # Blue, Orange, Green

    # 🧱 Draw bar plot
    splot = sns.barplot(
        data=df,
        x='LLMs', y='Values', hue='Category',
        palette=palette_colors, edgecolor='black', linewidth=0.7
    )

    # 🧩 Annotate bars with vertical values
    for bar in splot.patches:
        height = bar.get_height()
        if not np.isnan(height):
            x = bar.get_x() + bar.get_width() / 2
            splot.annotate(f'{height:.1f}',
                          xy=(x, height),
                          xytext=(0, 5),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8, rotation=90, fontweight='bold')

    # 🪶 Axis labels and limits
    ax.set_xlabel('')
    ax.set_ylabel('F1-Score (%)', fontsize=10)
    ax.set_ylim(y_limit)


    # 🧭 Customize axes and style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("gray")
    ax.spines['bottom'].set_color("gray")
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    # 📊 X-axis labels
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)

    # 🧭 Legend style
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        fontsize=9,
        frameon=True
    )

    plt.tight_layout()

    # 💾 Save the figure if needed
    if save:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f'{plt_name}.png'),
                    dpi=500, bbox_inches='tight')

    plt.show()

# classwise performance
classwise_plot_fake(df_fz,'4_fake_classwise_zero',y_limit=[30,100],save = True)

# classwise performance
classwise_plot_fake(df_ff,'4_fake_classwise_few',y_limit=[40,100],save = True)

## Confusion matrix function
def con_mat_fake(cm, class_names,model_name):
    plt.figure(figsize=(4, 3))  # 🔹 Bigger and clearer
    ax = plt.subplot()

    # 🔹 Use dark blue color map ('Blues' similar to your provided image)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        annot_kws={"size": 12, "weight": "bold", "color": "white"},
        square=True,
        cbar=True,
        linewidths=0.4,
        linecolor='white'
    )

    # 🔹 Label settings
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=6)
    ax.set_ylabel("True label", fontsize=11, labelpad=6)

    ax.xaxis.set_ticklabels(class_names, rotation=0, fontsize=10, weight='bold')
    ax.yaxis.set_ticklabels(class_names, rotation=0, fontsize=10, weight='bold')

    # 🔹 Adjust layout
    plt.tight_layout()

    # 🔹 Save high-quality figure
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"{model_name}.png"), dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()



## zero shot worst model
v_model = confusion_matrix(zero_fake['label'],zero_fake['mistral-7B'])
class_names =['Fake','Real']
con_mat_fake(v_model, class_names,"4_fake_mistral_conf_zero_worst_model")
#v_model

## zero shot best model
v_model = confusion_matrix(zero_fake['label'],zero_fake['gemma2-27B'])
class_names =['Fake','Real']
con_mat_fake(v_model, class_names,"4_fake_gemma_conf_zero_best_model")
#v_model

## few shot worst model

v_model = confusion_matrix(few_fake['label'],few_fake['mistral-7B'])
class_names =['Fake','Real']
con_mat_fake(v_model, class_names,"4_fake_mistral_conf_few_worst_model")
#v_model

## few shot best model
v_model = confusion_matrix(few_fake['label'],few_fake['gemma2-27B'])
class_names =['Fake','Real']
con_mat_fake(v_model, class_names,"4_fake_gemma_conf_few_best_model")
#v_model

## sentiment analysis
sz = [('llama32-3B', '51.83'),('mistral-7B', '52.25'), ('deepseek-8B', '52.15'), ('phi4-14B', '60.28'), ('gemma2-27B', '61.54'), ('qwen-72B', '59.32')]
sf = [('llama32-3B', '57.19'),('mistral-7B', '49.71'), ('deepseek-8B', '51.71'), ('phi4-14B', '58.88'), ('gemma2-27B', '63.00'), ('qwen-72B', '61.30')]
# emotion recognition
ez = [('llama32-3B', '44.28'),('mistral-7B', '33.89'), ('deepseek-8B', '42.14'),('phi4-14B', '56.96'),('gemma2-27B', '58.40'), ('qwen-72B', '59.44')]
ef = [('llama32-3B', '47.12'), ('mistral-7B', '46.41'), ('deepseek-8B', '43.41'),('phi4-14B', '54.56'),('gemma2-27B', '60.27'),('qwen-72B', '65.86')]
# hate speech
hz = [('llama32-3B', '54.22'),('mistral-7B', '64.19'), ('deepseek-8B', '56.40'), ('phi4-14B', '72.17'),('gemma2-27B', '76.11'),('qwen-72B', '79.88')]
hf = [('llama32-3B', '62.85'),('mistral-7B', '63.07'), ('deepseek-8B', '61.47'),('phi4-14B', '70.56'),('gemma2-27B', '73.39'),('qwen-72B', '78.25')]
# fake news
fz = [('llama32-3B', '77.72'),('mistral-7B', '55.45'),('deepseek-8B', '72.39'),('phi4-14B', '76.91'),('gemma2-27B', '84.33'), ('qwen-72B', '82.91')]
ff = [('llama32-3B', '66.27'),('mistral-7B', '58.93'), ('deepseek-8B', '65.59'), ('phi4-14B', '81.48'), ('gemma2-27B', '84.17'),('qwen-72B', '83.68')]

# Combine your data
data = {
    ('Sentiment', 'zero-shot'): sz,
    ('Sentiment', 'few-shot'): sf,
    ('Emotion', 'zero-shot'): ez,
    ('Emotion', 'few-shot'): ef,
    ('Hate Speech', 'zero-shot'): hz,
    ('Hate Speech', 'few-shot'): hf,
    ('Fake News', 'zero-shot'): fz,
    ('Fake News', 'few-shot'): ff,
}

# Build a flat list of records
records = []
for (task, setting), values in data.items():
    for llm, score in values:
        records.append({
            "Dataset": task,
            "Setting": setting,
            "LLM": llm,
            "F1-Score": float(score)
        })

df = pd.DataFrame(records)
print(df.head())


df

import matplotlib.pyplot as plt
import seaborn as sns

# Create the plot
plt.figure(figsize=(5, 5))
sns.stripplot(data=df[df["Setting"] == "zero-shot"], x="Dataset", y="F1-Score", hue="LLM", 
              jitter=2, dodge=1.5, size=10, palette="tab10",
              edgecolor="black",  # Border for each point
              linewidth=1)
plt.title("Zero-shot Performance")
plt.xlabel('Tasks')
plt.ylabel("F1-Score (%)")

# Remove the figure border box
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)

# Relabel the legends
new_labels = ['LLaMA-3.2-3B', 'Mistral-V3-7B', 'DeepSeek-R1-8B', 'Phi-4-14B', 'Gemma-2-27B', 'Qwen-2.5-72B']
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=new_labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
           shadow=False, ncol=3, fontsize=8, handletextpad=0.5, columnspacing=0.8, handlelength=1.5, handleheight=1.2)


# Adjust layout and display the plot
plt.tight_layout()
plt.savefig(os.path.join(results_dir,"5_zero_shot.png"),bbox_inches='tight',dpi =500)
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Create the plot
plt.figure(figsize=(5, 5))
sns.stripplot(data=df[df["Setting"] == "few-shot"], x="Dataset", y="F1-Score", hue="LLM", 
              jitter=True, dodge=True, size=10, palette="tab10",
              edgecolor="black",  # Border for each point
              linewidth=1)
plt.title("Few-shot Performance")
plt.xlabel('Tasks')
plt.ylabel("F1-Score (%)")

# Remove the figure border box
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['left'].set_visible(False)
#plt.gca().spines['bottom'].set_visible(False)

# Relabel the legends
new_labels = ['LLaMA-3.2-3B', 'Mistral-V3-7B', 'DeepSeek-R1-8B', 'Phi-4-14B', 'Gemma-2-27B', 'Qwen-2.5-72B']
handles, _ = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=new_labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
           shadow=False, ncol=3, fontsize=8, handletextpad=0.5, columnspacing=0.8, handlelength=1.5, handleheight=1.2)


# Adjust layout and display the plot
plt.tight_layout()
plt.savefig(os.path.join(results_dir,"6_few_shot.png"),bbox_inches='tight',dpi =500)
plt.show()



from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ------------------- CONFIG -------------------
llm_names = ['llama32-3B', 'mistral-7B', 'deepseek-8B', 'phi4-14B', 'gemma2-27B', 'qwen-72B']
datasets = ['senti', 'emo', 'hate', 'fake']
modes = ['zero_shot', 'few_shot']

current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
results_dir = os.path.join(root_dir, 'Results')
os.makedirs(results_dir, exist_ok=True)

# Dataset class labels
class_labels = {
    'senti': ['negative', 'neutral', 'positive'],
    'emo': ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'],
    'hate': ['hate', 'not_hate'],
    'fake': ['fake', 'real']
}

# ------------------- ROC FUNCTION -------------------
def plot_roc_auc(df, dataset_name, mode, class_names, show=True):
    plt.figure(figsize=(5.5, 4.5))
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(df['label'])
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    for model in llm_names:
        if model not in df.columns:
            continue

        y_pred = encoder.transform(df[model])
        y_pred_bin = label_binarize(y_pred, classes=np.unique(y_true))

        # Compute ROC curve and AUC
        roc_auc_list = []
        for i in range(y_true_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc_list.append(auc(fpr, tpr))

        mean_auc = np.mean(roc_auc_list)
        plt.plot(fpr, tpr, lw=1.5, label=f"{display_names[model]} (AUC={mean_auc:.2f})")

    # Plot format
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    dataset_full_names = {
        'senti': 'Sentiment Analysis',
        'emo': 'Emotion Recognition',
        'hate': 'Hate Speech Detection',
        'fake': 'Fake News Detection'
    }
    title_dataset = dataset_full_names.get(dataset_name, dataset_name.upper())
    plt.title(f"ROC–AUC: {title_dataset} ({mode.replace('_', ' ').title()})", fontsize=11)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(results_dir, f"roc_{dataset_name}_{mode}.png")
    plt.savefig(save_path, dpi=500, bbox_inches='tight')

    # Show in notebook
    if show:
        plt.show()
    else:
        plt.close()

# ------------------- GENERATE & DISPLAY ALL -------------------
for dataset in datasets:
    for mode in modes:
        file_path = os.path.join(results_dir, f"{dataset}_{mode}.xlsx")
        if os.path.exists(file_path):
            print(f"✅ Processing {dataset.upper()} - {mode.replace('_',' ').title()}")
            df = pd.read_excel(file_path)
            plot_roc_auc(df, dataset, mode, class_labels[dataset], show=True)
        else:
            print(f"⚠️ Missing file: {file_path}")

