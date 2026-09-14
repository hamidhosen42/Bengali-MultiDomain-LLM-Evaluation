# Evaluating the Effectiveness of Open-Weight LLMs for Multi-Domain Bengali Text Classification with Zero-Shot and Few-Shot Prompting



### ABSTRACT

Large Language Models (LLMs) have advanced multilingual NLP significantly, yet their effectiveness for Bengali, a globally important but low-resource language, remains understudied due to limited linguistic resources and the absence of a standardized evaluation benchmark. This study aims to address the underrepresentation of Bengali text classification by leading a systematic evaluation. It involves assessing multiple open-weight LLMs on Bengali text classification tasks and observing their limitations and regular error patterns. The procedure consists of six open-weight LLMs, which include LLaMA-3.2-3B, Mistral-V3-7B, DeepSeek-R1-8B, Phi-4-14B, Gemma-2-27B, and Qwen-2.5-72B, and are evaluated using four critical text classification tasks in Bengali, namely sentiment analysis, emotion recognition, hate-speech detection, and fake-news classification. Furthermore, two prompting methods are compared in the context of zero-shot and few-shot prompting using both the vLLM inference engine and Hugging Face Transformers pipelines to investigate the adaptability of the models in resource-limited settings. Across all four tasks, the two largest models (Qwen-2.5-72B and Gemma-2-27B) obtained the highest scores, although the ranking was not strictly monotonic in parameter count and, because each scale is represented by a single model family, the design cannot attribute these differences to model size alone. When training in zero-shot, Qwen-2.5-72B showed the best performance at 79.90% accuracy and 79.88% Macro-F1 score, showing amazing accuracy on hate-speech detection. In the case of fake news, Gemma-2-27B performed best at 84.45% accuracy and 84.33% Macro-F1 score. With a few-shot set-up, Gemma-2-27B and Qwen-2.5-72B demonstrated a high level of performance in terms of Macro-F1 score, reaching 84.17% for fake-news detection and 66.80% for emotion recognition, respectively. Paired multi-model tests (Cochran's Q with Holm-corrected McNemar post-hoc comparisons) confirmed significant differences among the six models across all tasks and prompting settings, whereas the effect of five-shot prompting, obtained with a single fixed demonstration set, was significant for only 4 of 24 model–task combinations after correction and never for the three largest models. The findings characterise benchmark classification performance under the evaluated settings rather than real-world effectiveness, and indicate that model scale, multilingual pre-training, and in-context demonstrations are jointly associated with, but cannot be causally isolated as drivers of, performance on Bengali text classification.


<p align="center">
    <a><img src="llm-perf.jpg" width="100%"></a>  <br>
</p>
<p>
<em> LLMs performance with two prompt settings across various classification tasks including Sentiment Analysis, Emotion Recognition, Hate Speech Detection, and Fake News Detection.</em></p>


# Instructions

### 1. Prerequisites
- **Python**: Ensure you have `Python 3.10.x` installed.
- **Conda** (Recommended): For managing virtual environments to avoid dependency conflicts.

### 2. Clone the Repository
Clone this repository to your local machine and navigate into the directory:
```bash
git clone https://github.com/hamidhosen42/Bengali-MultiDomain-LLM-Evaluation.git
cd Bengali-MultiDomain-LLM-Evaluation
```

### 3. Create a Virtual Environment
Create and activate an isolated environment using Conda:
```bash
conda create -n NLU python=3.10.12 -y
conda activate NLU
```

### 4. Install Dependencies
Install the required base libraries:
```bash
pip install -r requirements.txt
```

**Notes on Specific Dependencies:**
- **Inference Engine:** This repository utilizes both [vLLM](https://docs.vllm.ai/en/stable/index.html) and the Hugging Face `pipeline` for LLM inference. To use vLLM, install it according to the [official vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation.html).
- **Quantized Models:** If you are evaluating AWQ quantized models (such as `Qwen-2.5-72B`, evaluated in its AWQ form), make sure to install `gptqmodel` and a compatible `torch` version to avoid runtime errors:
  ```bash
  pip install gptqmodel torch
  ```

## Experimental Hardware Setup

To ensure reproducibility, experiments were conducted across two distinct environments based on model memory constraints:
- **Local Workstation:** A single NVIDIA RTX 4060 (8GB VRAM) was used to evaluate the smaller models.
- **Cloud Environment:** A single NVIDIA RTX PRO 6000 (96GB VRAM) hosted on RunPod was utilized exclusively for executing and verifying the Qwen-2.5-72B (AWQ) model. No multi-GPU clusters were used.

## Evaluated LLMs

To ensure full transparency and reproducibility, we utilized the official model weights hosted on the Hugging Face Hub. Below are the exact repository identifiers used for evaluation:

- **LLaMA-3.2-3B:** [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Mistral-V3-7B:** [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **DeepSeek-R1-8B:** [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- **Phi-4-14B:** [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4)
- **Gemma-2-27B:** [`google/gemma-2-27b-it`](https://huggingface.co/google/gemma-2-27b-it)
- **Qwen-2.5-72B (AWQ):** [`Qwen/Qwen2.5-72B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ)

Prompts for each task are organized in the `Prompts` folder.



## LLM Inference (Examples)

Navigate to the `Scripts` directory before running the inference scripts:
```bash
cd Scripts
```

### Example 1: Llama-3.2-3B
To evaluate the **Llama-3.2-3B** model with **Zero-Shot** prompting on the Emotion dataset:
```bash
python zero_few_shot.py \
  --llm_id meta-llama/Llama-3.2-3B-Instruct \
  --llm_name llama32-3B \
  --dataset_name emo \
  --prompt_type zero
```

### Example 2: Mistral-V3-7B
To evaluate the **Mistral-V3-7B** model with **Few-Shot** prompting on the Sentiment dataset:
```bash
python zero_few_shot.py \
  --llm_id mistralai/Mistral-7B-Instruct-v0.3 \
  --llm_name mistral-7B \
  --dataset_name senti \
  --prompt_type few
```

### Example 3: DeepSeek-R1-8B
To evaluate the **DeepSeek-R1-8B** model with **Zero-Shot** prompting on the Fake News dataset:
```bash
python zero_few_shot.py \
  --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --llm_name deepseek-8B \
  --dataset_name fake \
  --prompt_type zero
```

### Example 4: Phi-4-14B
To evaluate the **Phi-4-14B** model with **Few-Shot** prompting on the Hate Speech dataset:
```bash
python zero_few_shot.py \
  --llm_id microsoft/phi-4 \
  --llm_name phi4-14B \
  --dataset_name hate \
  --prompt_type few
```

### Example 5: Gemma-2-27B
To evaluate the **Gemma-2-27B** model with **Zero-Shot** prompting on the Emotion dataset:
```bash
python zero_few_shot.py \
  --llm_id google/gemma-2-27b \
  --llm_name gemma2-27B \
  --dataset_name emo \
  --prompt_type zero
```

### Example 6: Qwen-2.5-72B-AWQ
To evaluate the **Qwen-2.5-72B-AWQ** model with **Few-Shot** prompting on the Hate Speech dataset:
```bash
python zero_few_shot.py \
  --llm_id Qwen/Qwen2.5-72B-Instruct-AWQ \
  --llm_name qwen-72B \
  --dataset_name hate \
  --prompt_type few
```

### Running All Models and Datasets

To systematically run a model across all datasets and settings, you can use a simple bash loop. For example, to run the zero-shot setting for all four datasets using Llama 3.2:

```bash
for dataset in emo hate senti fake; do
  python zero_few_shot.py \
    --llm_id meta-llama/Llama-3.2-3B-Instruct \
    --llm_name llama32-3B \
    --dataset_name $dataset \
    --prompt_type zero
done
```
Modify the `--llm_id`, `--llm_name`, and `--prompt_type` accordingly for other models.

### Arguments

- `--llm_id`: The Hugging Face model ID (e.g., `Qwen/Qwen2.5-72B-Instruct-AWQ`).
- `--llm_name`: A short identifier used for saving results (e.g., `qwen-72B`).
- `--dataset_name`: The dataset to evaluate on. Options: `senti`, `emo`, `hate`, or `fake`.
- `--prompt_type`: The prompting technique to use. Options: `zero` or `few`.

You will get an excel file in **Results/** folder that store the responses for the corresponding LLM.

*The `plot-notebook.ipynb`* file contains the code for the result visualization.

---

## Statistical Significance Testing

For the two top-performing models (Qwen-2.5-72B and Gemma-2-27B) we performed **paired permutation resampling tests** (1,000 iterations). Because 16 tests are run at once, the resulting p-values are also reported after **Holm–Bonferroni correction** (see `Results/qwen_gemma_permutation_holm.csv`). In addition, all six models are compared together with a **paired multi-model analysis** (Cochran's Q test followed by Holm-corrected pairwise exact McNemar tests), and the zero-shot vs. five-shot predictions of each model are compared with the same McNemar test (see the section below). This non-parametric approach does not assume any underlying data distribution and is highly robust for evaluating differences in paired classification tasks across multiple metrics like Accuracy and Macro F1 score.

### Running the Test
You can run the significance testing script directly from the root directory:
```bash
python evaluate_significance.py
```
This will output a `significance_test_results.csv` and `significance_tests_raw.xlsx` inside the `Results/` folder.

### Significance Results Table
The following table reports the raw $p$-values of the permutation test and the Holm-adjusted $p$-values (family of 16 tests, $\alpha = 0.05$):

| Dataset | Prompt Setting | Accuracy $p$ (raw) | Accuracy $p$ (Holm) | Macro F1 $p$ (raw) | Macro F1 $p$ (Holm) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Emotion** | Zero-shot | 1.000 | 1.000 | 0.943 | 1.000 |
| **Emotion** | Few-shot | 0.003 | 0.039* | 0.016 | 0.160 |
| **Fake News** | Zero-shot | 0.475 | 1.000 | 0.306 | 1.000 |
| **Fake News** | Few-shot | 0.876 | 1.000 | 0.658 | 1.000 |
| **Hate Speech** | Zero-shot | 0.007 | 0.077 | 0.004 | 0.048* |
| **Hate Speech** | Few-shot | 0.001 | 0.016* | <0.001 | 0.016* |
| **Sentiment** | Zero-shot | <0.001 | 0.016* | 0.101 | 0.707 |
| **Sentiment** | Few-shot | 0.023 | 0.207 | 0.055 | 0.440 |

*\* denotes statistical significance at $\alpha = 0.05$ after Holm correction.*

## Multi-Model Paired Analysis (all six models)

Cochran's Q test checks whether the six models have the same accuracy on the same test items; the 15 model pairs are then compared with the exact McNemar test with Holm correction. Run:
```bash
python multi_model_significance.py
```
Outputs in `Results/`: `cochran_q_results.csv`, `pairwise_mcnemar_holm.csv`, `fewshot_effect_mcnemar.csv`, `qwen_gemma_permutation_holm.csv`, `f1_confidence_intervals_all_models.csv`.

| Dataset | Prompt Setting | Cochran's Q (df = 5) | $p$ | Significant pairs (Holm) | Pairs not significantly different |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentiment** | Zero-shot | 133.1 | <0.001 | 10/15 | LLaMA–Mistral; LLaMA–DeepSeek; Mistral–DeepSeek; Phi-4–Gemma; Phi-4–Qwen |
| **Sentiment** | Few-shot | 193.5 | <0.001 | 10/15 | LLaMA–Phi-4; LLaMA–Qwen; Mistral–DeepSeek; Phi-4–Qwen; Gemma–Qwen |
| **Emotion** | Zero-shot | 263.7 | <0.001 | 11/15 | LLaMA–DeepSeek; Phi-4–Gemma; Phi-4–Qwen; Gemma–Qwen |
| **Emotion** | Few-shot | 130.9 | <0.001 | 12/15 | LLaMA–Mistral; LLaMA–DeepSeek; Mistral–DeepSeek |
| **Hate Speech** | Zero-shot | 282.7 | <0.001 | 14/15 | LLaMA–DeepSeek |
| **Hate Speech** | Few-shot | 159.8 | <0.001 | 12/15 | LLaMA–Mistral; LLaMA–DeepSeek; Mistral–DeepSeek |
| **Fake News** | Zero-shot | 140.1 | <0.001 | 11/15 | LLaMA–DeepSeek; LLaMA–Phi-4; DeepSeek–Phi-4; Gemma–Qwen |
| **Fake News** | Few-shot | 202.5 | <0.001 | 10/15 | LLaMA–DeepSeek; Mistral–DeepSeek; Phi-4–Gemma; Phi-4–Qwen; Gemma–Qwen |

**Effect of five-shot prompting** (change in accuracy vs. zero-shot, percentage points, with Holm-adjusted McNemar $p$ over 24 tests; * = significant at 0.05):

| Model | Sentiment | Emotion | Hate Speech | Fake News |
| :--- | :--- | :--- | :--- | :--- |
| **LLaMA-3.2-3B** | +4.9 (0.009)* | +0.5 (1.000) | +7.3 (0.015)* | -8.5 (0.007)* |
| **Mistral-V3-7B** | -3.4 (0.074) | +9.3 (0.001)* | -1.6 (1.000) | +2.0 (1.000) |
| **DeepSeek-R1-8B** | -0.2 (1.000) | +0.6 (1.000) | +2.6 (1.000) | -5.1 (0.751) |
| **Phi-4-14B** | -1.5 (0.538) | -3.8 (0.100) | -1.5 (1.000) | +4.3 (0.052) |
| **Gemma-2-27B** | +0.5 (1.000) | -0.6 (1.000) | -2.2 (0.192) | +0.0 (1.000) |
| **Qwen-2.5-72B** | +2.2 (0.188) | +4.0 (0.179) | -1.5 (0.980) | +0.8 (1.000) |

## Quantitative Error Analysis

Confusion-pair shares, false-negative/false-positive asymmetry for hate speech, a negation-marker proxy for sentiment/emotion, and surface cues (numerals, length, quotation marks) of missed fake articles are computed from the saved predictions:
```bash
python quantitative_error_analysis.py
```
Outputs in `Results/`: `error_confusion_pairs.csv`, `error_hate_fn_fp.csv`, `error_negation_proxy.csv`, `error_fake_surface_cues.csv`.

---

### 95% Confidence Intervals for Macro F1 Score
To account for the varying test set sizes across the tasks, we computed the 95% Confidence Intervals (CI) for the Macro F1 score of all six models using bootstrap resampling (1,000 iterations).

| Dataset (Test Size) | Prompt Setting | LLaMA-3.2-3B | Mistral-V3-7B | DeepSeek-R1-8B | Phi-4-14B | Gemma-2-27B | Qwen-2.5-72B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Emotion (625)** | Zero-shot | 45.88 (41.90, 49.79) | 36.08 (32.19, 39.88) | 44.08 (40.07, 47.73) | 59.89 (56.28, 63.28) | 61.80 (57.95, 65.39) | 61.91 (58.08, 65.96) |
| **Emotion (625)** | Few-shot | 46.35 (41.84, 50.22) | 45.18 (40.83, 49.18) | 45.85 (42.40, 49.47) | 57.90 (54.23, 61.03) | 63.01 (59.59, 66.37) | 66.80 (63.11, 70.17) |
| **Fake News (508)** | Zero-shot | 77.72 (74.21, 81.10) | 55.45 (50.79, 59.82) | 72.39 (68.50, 76.47) | 76.91 (73.09, 80.57) | 84.33 (80.87, 87.37) | 82.91 (79.57, 86.11) |
| **Fake News (508)** | Few-shot | 66.27 (61.54, 70.49) | 58.93 (54.67, 63.25) | 65.59 (61.31, 69.80) | 81.48 (77.63, 84.80) | 84.17 (80.98, 87.34) | 83.68 (80.28, 86.80) |
| **Hate Speech (1000)** | Zero-shot | 54.22 (51.01, 57.60) | 64.19 (61.03, 67.09) | 56.40 (53.60, 59.61) | 72.17 (69.50, 74.90) | 76.11 (73.38, 78.71) | 79.88 (77.49, 82.28) |
| **Hate Speech (1000)** | Few-shot | 62.85 (59.79, 65.74) | 63.07 (59.89, 66.10) | 61.47 (58.45, 64.52) | 70.56 (67.69, 73.32) | 73.39 (70.56, 76.04) | 78.25 (75.65, 80.67) |
| **Sentiment (1586)** | Zero-shot | 49.62 (47.17, 52.03) | 50.20 (47.73, 52.66) | 49.90 (47.38, 52.28) | 57.73 (55.26, 60.06) | 58.66 (56.38, 61.25) | 56.84 (54.45, 59.22) |
| **Sentiment (1586)** | Few-shot | 54.61 (52.31, 57.06) | 48.09 (45.68, 50.56) | 49.47 (47.13, 51.82) | 56.44 (53.98, 58.81) | 60.68 (58.37, 63.18) | 58.82 (56.45, 61.11) |

You can generate these confidence intervals by running `python calculate_ci.py` (Qwen-2.5-72B and Gemma-2-27B) or `python multi_model_significance.py` (all six models).

---

## Acknowledgments
The authors deeply appreciate the **Centre for Advanced Analytics, Multimedia University**, Dr. Khair Razlan Othman, and Riadul Islam Rabbi for their important support and resources during this study. The cooperative environment and the knowledge provided by these institutions have been crucial to the accomplishment of this study's effective conclusion.

## Citation
If you find this repository or our study useful for your research, please consider citing our paper:
```bibtex
@article{hosen2025evaluating,
  title={Evaluating the effectiveness of open-weight LLMs for multi-domain Bengali text classification with zero-shot and few-shot prompting},
  author={Hosen, Md. Hamid and Nawar, Sadia and Chowdhury, Rituparna and Uddin, Altaf and Rabbi, Riadul Islam and Othman, Khair Razlan Bin and Symon, Nurul Karim},
  journal={PLOS ONE},
  year={2025}
}
```
