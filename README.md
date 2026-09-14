# Evaluating the Effectiveness of Open-Source LLMs for Multi-Domain Bengali Text Classification with Zero-Shot and Few-Shot Prompting



### ABSTRACT

Large Language Models (LLMs) have shown remarkable ability to process multilingual texts, while Bengali is still an underrepresented language. This is due to the unavailability of large linguistic resources and the lack of a standard benchmarking framework, which has yet to be established. In this study, six open-source LLMs, which include LLaMA-3.2-3B, Mistral-V3-7B, DeepSeek-R1-8B, Phi-4-14B, Gemma-2-27B, and Qwen-2.5-72B, are evaluated using four critical text classification tasks in Bengali, such as sentiment analysis, emotion recognition, hate-speech detection, and fake-news classification. We compared two prompting methods in the context of zero-shot and few-shot prompting using the vLLM inference engine and Hugging Face pipelines to investigate the adaptability of the model in resource-limited settings. The measurements of performance were based on Accuracy (Acc.), Precision (P), Recall (R), and F1-score (F1) measures. The results of our study indicate that, regardless of the dataset, the bigger multilingual models always performed better in comparison with the smaller ones. When training in zero-shot, Qwen-2.5-72B showed the best performance at 79.9% accuracy and 79.88% F1-score, showing specific effectiveness on hate-speech detection. In the case of fake news, Gemma-2-27B performed best at 84.45% accuracy and 84.33% F1- score. With a few-shot set-up, Gemma-2-27B and Qwen-2.5-72B had a high level of performance in terms of F1-score, 84.17 and 65.86, respectively. It is worth noting that few-shot prompting increased Recall by an average of about 6%, which proves the usefulness of in-context learning in low-resource settings. This study highlights the possibilities of large multilingual models in enabling robust, scalable, and accessible Bengali natural language processing applications.


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

To ensure the robustness of our comparative evaluation between the top-performing models (Qwen-2.5-72B and Gemma-2-27B), we performed **paired permutation resampling tests** (1,000 iterations). This non-parametric approach does not assume any underlying data distribution and is highly robust for evaluating differences in paired classification tasks across multiple metrics like Accuracy and Macro F1 score.

### Running the Test
You can run the significance testing script directly from the root directory:
```bash
python evaluate_significance.py
```
This will output a `significance_test_results.csv` and `significance_tests_raw.xlsx` inside the `Results/` folder.

### Significance Results Table
The following table reports the exact $p$-values calculated using our test ($\alpha = 0.05$):

| Dataset | Prompt Setting | Accuracy $p$-value | Macro F1 $p$-value | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Emotion** | Zero-shot | 1.000 | 0.943 | No significant difference |
| **Emotion** | Few-shot | 0.003* | 0.016* | **Significant difference** |
| **Fake News** | Zero-shot | 0.475 | 0.306 | No significant difference |
| **Fake News** | Few-shot | 0.876 | 0.658 | No significant difference |
| **Hate Speech** | Zero-shot | 0.007* | 0.004* | **Significant difference** |
| **Hate Speech** | Few-shot | 0.001* | <0.001* | **Significant difference** |
| **Sentiment** | Zero-shot | <0.001* | 0.101 | Acc: **Significant**, F1: Not Sig. |
| **Sentiment** | Few-shot | 0.023* | 0.055 | Acc: **Significant**, F1: Marginal |

*\* denotes statistical significance at $\alpha = 0.05$.*

### 95% Confidence Intervals for Macro F1 Score
To account for the varying test set sizes across the tasks, we computed the 95% Confidence Intervals (CI) for the Macro F1 score using bootstrap resampling (1,000 iterations).

| Dataset (Test Size) | Prompt Setting | Qwen-72B F1 (95% CI) | Gemma2-27B F1 (95% CI) |
| :--- | :--- | :--- | :--- |
| **Emotion (625)** | Zero-shot | 61.91 (58.08, 65.96) | 61.80 (57.95, 65.39) |
| **Emotion (625)** | Few-shot | 66.80 (63.11, 70.17) | 63.01 (59.59, 66.37) |
| **Fake News (508)** | Zero-shot | 82.91 (79.57, 86.11) | 84.33 (80.87, 87.37) |
| **Fake News (508)** | Few-shot | 83.68 (80.28, 86.80) | 84.17 (80.98, 87.34) |
| **Hate Speech (1000)** | Zero-shot | 79.88 (77.49, 82.28) | 76.11 (73.38, 78.71) |
| **Hate Speech (1000)** | Few-shot | 78.25 (75.65, 80.67) | 73.39 (70.56, 76.04) |
| **Sentiment (1586)** | Zero-shot | 56.84 (54.45, 59.22) | 58.66 (56.38, 61.25) |
| **Sentiment (1586)** | Few-shot | 58.82 (56.45, 61.11) | 60.68 (58.37, 63.18) |

You can generate these confidence intervals by running:
```bash
python calculate_ci.py
```

---

## Acknowledgments
The authors deeply appreciate the **Centre for Advanced Analytics, Multimedia University**, Dr. Khair Razlan Othman, and Riadul Islam Rabbi for their important support and resources during this study. The cooperative environment and the knowledge provided by these institutions have been crucial to the accomplishment of this study's effective conclusion.

## Citation
If you find this repository or our study useful for your research, please consider citing our paper:
```bibtex
@article{hosen2025evaluating,
  title={Evaluating the effectiveness of open-source LLMs for multi-domain Bengali text classification with zero-shot and few-shot prompting},
  author={Hosen, Md. Hamid and Nawar, Sadia and Chowdhury, Rituparna and Uddin, Altaf and Rabbi, Riadul Islam and Othman, Khair Razlan Bin and Symon, Nurul Karim},
  journal={PLOS ONE},
  year={2025}
}
```
