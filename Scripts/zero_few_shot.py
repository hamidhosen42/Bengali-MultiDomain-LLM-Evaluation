## ================================================================
## Zero/Few-Shot Classification - NVIDIA GPU Optimized (4-bit + Offload)
## ================================================================
"""
This script attempts (in order):
  1) 4-bit (nf4) quantized load with device_map="auto" and CPU offload for large layers
  2) fallback to 8-bit quantized load
  3) final fallback to standard precision pipeline

Prereqs (in your hamid env):
  pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install -U transformers accelerate bitsandbytes
  (bitsandbytes requires CUDA 12+ which you're using)
"""

import os
# Prevent TF imports (avoids the Keras/tf issues)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"
# ensure visible device 0 (optional)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from transformers import set_seed, AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import torch, numpy as np, pandas as pd
import json, re, ast, time, gc, warnings, argparse
warnings.filterwarnings('ignore')

set_seed(42)

## ================================================================
## Path Organization
## ================================================================
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
dataset_dir = os.path.join(root_dir, 'Datasets')
prompts_dir = os.path.join(root_dir, 'Prompts')

## ================================================================
## Helper Functions
## ================================================================
def extract_first_dictionary(json_string):
    """Extract the first JSON-like dictionary from a text string."""
    try:
        cleaned_json_string = re.sub(r'#.*$', '', json_string, flags=re.MULTILINE).strip()
        json_match = re.search(r'\{.*?\}', cleaned_json_string, re.DOTALL)
        return json_match.group() if json_match else None
    except Exception as e:
        print(f"Error extracting first dictionary: {e}")
        return None

def extract_labels(item):
    """Extract the 'label' field from a JSON string."""
    try:
        parsed = json.loads(item)
        return parsed.get('label', item)
    except (json.JSONDecodeError, TypeError):
        return item

def prompts(dataset_name, prompt_type):
    """Load prompt templates."""
    with open(os.path.join(prompts_dir, f"{dataset_name}_{prompt_type}_prompt.txt"), "r", encoding="utf-8") as file:
        return file.read()

def generate_with_pipeline(llm, messages, max_retries=3):
    """Generate output using the Hugging Face pipeline with retries."""
    for attempt in range(max_retries):
        try:
            text = messages[-1]["content"]
            output = llm(text, max_new_tokens=256)
            # HuggingFace pipelines return list of dicts or pipeline-specific structure
            if isinstance(output, list) and isinstance(output[0], dict) and "generated_text" in output[0]:
                return output[0]["generated_text"]
            return str(output)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 20s...")
                time.sleep(20)
            else:
                print("Max retries reached. Skipping sample.")
                return ""

## ================================================================
## Response Generation
## ================================================================
def response_generation(dataset, dataset_name, prompt_type, llm_info):
    dataset_map = {
        'emo': 'Emotion Recognition',
        'senti': 'Sentiment Analysis',
        'hate': 'Hate Speech Detection',
        'fake': 'Fake News Detection'
    }

    print("--------------------------------------")
    print(f"{prompt_type.capitalize()} Shot {dataset_map[dataset_name]}:")
    print(f"LLM: {llm_info['llm_name']}")
    print("--------------------------------------")

    sentence_list, results = [], []

    for i in range(len(dataset)):
        sentence = dataset['sentence'][i]
        print(f"Sample: {i}")

        prompt_template = prompts(dataset_name, prompt_type)
        prompt = prompt_template.format(text=sentence)

        messages = [
            {"role": "system", "content": "You are a precise and efficient classification model."},
            {"role": "user", "content": prompt}
        ]

        output = generate_with_pipeline(llm_info['llm'], messages)
        try:
            response = extract_first_dictionary(output)
            label = extract_labels(response) if response else output
            print(label)
            results.append(label)
        except Exception as e:
            print(f"Decoding Error at sample {i}, saving raw response: {e}")
            print(output)
            results.append(output)

        sentence_list.append(sentence)
        time.sleep(1.2)

    results_dir = os.path.join(root_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f'{dataset_name}_{prompt_type}_shot.xlsx')

    if os.path.exists(output_file):
        data = pd.read_excel(output_file)
    else:
        data = pd.DataFrame({'sentence': sentence_list})

    data[llm_info['llm_name']] = results
    data.to_excel(output_file, index=False)
    print(f"✅ Results saved to: {output_file}")

## ================================================================
## Model loading helpers
## ================================================================
def try_load_4bit(model_id, dtype, offload_folder):
    """
    Attempt 4-bit NF4 load with auto device mapping and offload.
    Returns (model, tokenizer, note) or raises.
    """
    print("Attempting 4-bit (nf4) quantized load with device_map='auto' and offload...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # from_pretrained supports: device_map="auto", quantization_config=bnb_config,
    # and kwargs like offload_folder, offload_state_dict, and low_cpu_mem_usage
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",                # let accelerate / transformers split model across GPU/CPU
        offload_folder=offload_folder,    # temporary folder to offload weights if needed
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer, "4bit (nf4)"

def try_load_8bit(model_id, dtype):
    print("Attempting 8-bit quantized load (bitsandbytes)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer, "8bit"

def try_load_fp16_pipeline(model_id, device, dtype):
    print("Falling back to standard pipeline (fp16 if GPU, else fp32)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # pipeline will load model under the hood; prefer providing tokenizer only to let HF decide
    llm = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=dtype
    )
    return llm, tokenizer, "pipeline-fp"

## ================================================================
## Main
## ================================================================
def main(args):
    # Load dataset
    if args.dataset_name == 'emo':
        dataset = pd.read_excel(os.path.join(dataset_dir, 'emo_test.xlsx'))
    elif args.dataset_name == 'senti':
        dataset = pd.read_excel(os.path.join(dataset_dir, 'senti_test.xlsx'))
    elif args.dataset_name == 'hate':
        dataset = pd.read_excel(os.path.join(dataset_dir, 'hate_test.xlsx'))
    elif args.dataset_name == 'fake':
        dataset = pd.read_excel(os.path.join(dataset_dir, 'fake_test.xlsx'))
    else:
        raise ValueError("Invalid dataset name provided.")

    gc.collect()

    # Device detection
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"✅ CUDA available. GPU: {torch.cuda.get_device_name(0)}")
        # Use bfloat16 for 4-bit compute dtype; for pipeline dtype use torch.bfloat16 if supported
        dtype_for_model = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
        pipeline_dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
        device_for_pipeline = 0  # HF uses int for GPU device index in pipeline
    else:
        print("⚙ CUDA not available — running on CPU.")
        dtype_for_model = torch.float32
        pipeline_dtype = torch.float32
        device_for_pipeline = -1  # CPU

    model_id = args.llm_id
    offload_folder = os.path.join(root_dir, "offload_dir")
    os.makedirs(offload_folder, exist_ok=True)

    llm_obj = None
    tokenizer = None
    llm_name_note = None

    # Try 4-bit first (best chance to fit Zephyr-7B into 8GB)
    if gpu_available:
        try:
            model, tokenizer, llm_name_note = try_load_4bit(model_id, dtype_for_model, offload_folder)
            # when model loaded, create pipeline from model & tokenizer
            llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
            print(f"✅ Loaded model using: {llm_name_note}")
            llm_obj = llm
        except Exception as e4:
            print(f"⚠ 4-bit load failed: {e4}")
            # try 8-bit
            try:
                model, tokenizer, llm_name_note = try_load_8bit(model_id, dtype_for_model)
                llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
                print(f"✅ Loaded model using: {llm_name_note}")
                llm_obj = llm
            except Exception as e8:
                print(f"⚠ 8-bit load failed: {e8}")
                # fallback to pipeline-based load (might be large and CPU-heavy)
                try:
                    llm, tokenizer, llm_name_note = try_load_fp16_pipeline(model_id, device_for_pipeline, pipeline_dtype)
                    print(f"✅ Loaded model via pipeline fallback ({llm_name_note})")
                    llm_obj = llm
                except Exception as e_fp:
                    print(f"❌ Final fallback failed: {e_fp}")
                    raise RuntimeError("Could not load model in any mode. See errors above.") from e_fp
    else:
        # No GPU: just use pipeline (CPU)
        try:
            llm, tokenizer, llm_name_note = try_load_fp16_pipeline(model_id, device_for_pipeline, pipeline_dtype)
            print(f"✅ Loaded model via pipeline fallback ({llm_name_note})")
            llm_obj = llm
        except Exception as e_cpu:
            print(f"❌ CPU pipeline load failed: {e_cpu}")
            raise

    llm_info = {
        "tokenizer": tokenizer,
        "llm": llm_obj,
        "llm_name": args.llm_name
    }

    response_generation(dataset, args.dataset_name, args.prompt_type, llm_info)

## ================================================================
## Entry point
## ================================================================
if __name__ == "_main_":
    parser = argparse.ArgumentParser(description='Zero/Few Shot Classification - NVIDIA GPU Optimized (4-bit + Offload)')
    parser.add_argument('--llm_id', type=str, required=True, help='Model ID (e.g., HuggingFaceH4/zephyr-7b-beta)')
    parser.add_argument('--llm_name', type=str, required=True, help='Short model name (e.g., zephyr7b)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (emo/senti/hate/fake)')
    parser.add_argument('--prompt_type', type=str, required=True, help='Prompt type (zero/few)')
    args = parser.parse_args()
    main(args)