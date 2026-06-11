import os

filepath = r'e:\Evaluating the Effectiveness of Open-Source LLMs for Multi-Domain Bengali Text Classification with Zero-Shot and Few-Shot Prompting\Revised_Manuscript_with_Track_Changes___PLOS_ONE\plos_latex_template.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

replacements = [
    # 1. Neutral class underrepresented
    (
        r"consistently showing that separating neutral sentiment is still difficult in the absence of context examples.",
        r"consistently showing that separating neutral sentiment is still difficult in the absence of context examples. This difficulty is partially exacerbated by the fact that the neutral class is underrepresented in the dataset (361 samples compared to 654 positive and 571 negative)."
    ),
    # 1b. DeepSeek context
    (
        r"with substantial improvements for the neutral class observed across several models.",
        r"with substantial improvements for the neutral class observed across several models (including DeepSeek-R1-8B, which struggles with neutral sentiment under zero-shot conditions but benefits from few-shot examples)."
    ),
    # 2. Emotion classes overlap Qwen
    (
        r"Qwen-2.5-72B improves performance by 15\% in the Disgust class compared to zero-shot prompting, it results in a 5\% decrease in the Anger class.",
        r"Qwen-2.5-72B improves performance by 15\% in the Disgust class compared to zero-shot prompting, it results in a 5\% decrease in the Anger class. This trade-off suggests a potential semantic overlap between expressions of anger and disgust in the dataset, leading the model to occasionally misclassify nuanced emotional cues."
    ),
    # 3. Hate Speech Qwen
    (
        r"(e.g., Qwen-2.5-72B,",
        r"(unlike Qwen-2.5-72B,"
    ),
    # 3b. Hate Speech generalisation and languages
    (
        r"This disparity indicates that smaller-sized or less multilingual models do not generalise to socially sensitive language situations in Bengali very well.",
        r"This disparity indicates that smaller-sized or less multilingual models do not generalise to socially sensitive language situations in Bengali very well. Similar challenges have been observed in other low-resource languages, where limited representation in pretraining data restricts a model's ability to grasp context-dependent cultural expressions."
    ),
    # 4. Fake News LLaMA-3.2-3B and decrease typo
    (
        r"such as LLaMA-3.2-3B and DeepSeek-R1-8B obtain slight improvements",
        r"such as DeepSeek-R1-8B obtain slight improvements (while LLaMA-3.2-3B actually performs worse)"
    ),
    (
        r"shows a marginal decrease from 80.5\% to 81.2\% for the fake class and from 85.4\% to 86.2\% for the real class.",
        r"shows a marginal increase from 80.5\% to 81.2\% for the fake class and from 85.4\% to 86.2\% for the real class."
    ),
    # 5. Cautious prediction bias example
    (
        r"This reflects a cautious prediction bias, where the models avoid labelling a text as hateful unless explicit aggression is present.",
        r"This reflects a cautious prediction bias, where the models avoid labelling a text as hateful unless explicit aggression is present. For example, sentences expressing disagreement without vulgarity are often misclassified as non-hate by these models."
    ),
    # 6. ROC AUC multiclass
    (
        r"In ROC curves, the true positive rate is plotted against the false positive rate across multiple thresholds to highlight the confidence and separation capabilities of the models for the positive and negative classes.",
        r"In ROC curves, the true positive rate is plotted against the false positive rate across multiple thresholds. For multiclass classification (e.g., Sentiment Analysis, Emotion Recognition), the One-vs-Rest (OvR) strategy is employed to calculate the area under the ROC curve (AUC), where each class is iteratively treated as the positive class against all others."
    ),
    # 7. Discussion Multilingual Pretraining
    (
        r"On the other hand, Qwen-2.5-72B and Gemma-2-27B always work well since they have superior multilingual pretraining, cover more tokens, and represent Indic scripts better.",
        r"On the other hand, Qwen-2.5-72B and Gemma-2-27B consistently perform well. Qwen-2.5-72B benefits from superior multilingual pretraining and better representation of Indic scripts. While Gemma-2-27B was trained primarily on English data and not explicitly optimized for multilingual state-of-the-art performance, its sheer scale and architectural efficiency still allow it to perform robustly on Bengali tasks."
    ),
    # 8. Limitations rewording
    (
        r"Details of the limitation of the workIn this work, the",
        r"In this work, the"
    ),
    # 9. Typos
    (r"across four Bengali test classification", r"across four Bengali text classification"),
    (r"TinyLLAM", r"TinyLLaMA"),
    (r"resources for NLP.", r"resources for NLP"),
    (r"In this research experiment was conducted on six", r"In this research, experiment was conducted on six"),
    (r"set g refers to a set of keys and 305 values, where g < h.", r"set of keys and values."),
    (r"reinforming appropriate language generation", r"reinforcement of appropriate language generation"),
    (r"a reference polity", r"a reference policy"),
    (r"contemporary evaluation metrics: Accuracy,", r"established evaluation metrics: Accuracy,"),
    
    # 10. References
    (
        r"In 2023, the authors in [12] presented the Transformer decoder architecture",
        r"In 2017, Vaswani et al. introduced the Transformer architecture \cite{vaswani2017attention}, which later popularized the decoder-only paradigm used by modern LLMs \cite{brown2020language}"
    ),
    (
        r"Joboji",
        r"Authors"
    )
]

count = 0
for search_str, replace_str in replacements:
    if search_str in text:
        text = text.replace(search_str, replace_str)
        count += 1
    else:
        print(f"Warning: Could not find '{search_str[:30]}...' in the manuscript.")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Manuscript updated successfully. Replaced {count} instances.")
