# zero_few_shot.py
import os
import re
import time
import json
import gc
import argparse
import warnings
import sys
import math

import numpy as np
import pandas as pd
import torch
from transformers import set_seed, AutoTokenizer

warnings.filterwarnings("ignore")
set_seed(42)

# ----------------
# Path Organization
# ----------------
current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
dataset_dir = os.path.join(root_dir, "Datasets")
prompts_dir = os.path.join(root_dir, "Prompts")

# ----------------
# Utility helpers
# ----------------
def extract_first_dictionary(json_string):
    """Extract first {...} dictionary-like substring from a string."""
    try:
        if not isinstance(json_string, str):
            json_string = str(json_string)
        cleaned_json_string = re.sub(r"#.*$", "", json_string, flags=re.MULTILINE).strip()
        json_match = re.search(r"\{.*?\}", cleaned_json_string, re.DOTALL)
        if json_match:
            return json_match.group()
        return None
    except Exception as e:
        print(f"Error extracting first dictionary: {e}")
        return None

def extract_labels(item):
    """Parse JSON dictionary string and return 'label' field (fallback to raw item)."""
    try:
        parsed = json.loads(item)
        return parsed.get("label", parsed)
    except (json.JSONDecodeError, TypeError):
        return item

def prompts(dataset_name, prompt_type):
    file_path = os.path.join(prompts_dir, f"{dataset_name}_{prompt_type}_prompt.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fh:
        return fh.read()

# ----------------
# Generation wrappers (unified)
# ----------------
def generate_with_pipeline(pipe, messages, max_retries=3, max_new_tokens=256):
    """Generate text using a HF transformers pipeline."""
    if isinstance(messages, list):
        prompt_text = "\n".join([m.get("content", "") for m in messages])
    else:
        prompt_text = str(messages)

    for attempt in range(max_retries):
        try:
            out = pipe(prompt_text, max_new_tokens=max_new_tokens)
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"]
                if isinstance(first, str):
                    return first
                return str(first)
            return str(out)
        except Exception as e:
            print(f"Pipeline attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return ""

def generate_with_model_tokenizer(model, tokenizer, device, messages, max_new_tokens=256, temperature=0.0):
    """
    Generate using an explicit model + tokenizer.
    - model: an instance of AutoModelForCausalLM (moved to device)
    - tokenizer: corresponding tokenizer
    - device: torch.device (e.g., torch.device('cuda'), dml device, or cpu)
    """
    if isinstance(messages, list):
        prompt_text = "\n".join([m.get("content", "") for m in messages])
    else:
        prompt_text = str(messages)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=False)
    # move inputs to device; for torch-directml the device move may be different but `.to(device)` usually works
    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        # fallback: maybe device not supported for tensors; keep on cpu
        inputs = inputs

    # generation params
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if hasattr(model.config, "use_cache"):
        gen_kwargs["use_cache"] = True

    # generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # decode
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# ----------------
# Core response generation
# ----------------
def response_generation(dataset, dataset_name, prompt_type, llm_info):
    dataset_map = {
        "emo": "Emotion Recognition",
        "senti": "Sentiment Analysis",
        "hate": "Hate Speech Detection",
        "fake": "Fake News Detection",
    }

    print("--------------------------------------")
    print(f"{prompt_type.capitalize()} Shot {dataset_map.get(dataset_name, dataset_name)}:")
    print(f"LLM: {llm_info['llm_name']} (mode={llm_info['mode']})")
    print("--------------------------------------")

    sentence_list = []
    results = []

    total = len(dataset)
    for i in range(total):
        sentence = dataset["sentence"].iloc[i] if "sentence" in dataset.columns else dataset.iloc[i, 0]
        print(f"Sample: {i+1}/{total}")

        prompt_template = prompts(dataset_name, prompt_type)
        prompt = prompt_template.format(text=sentence)

        if any(x in llm_info["llm_name"] for x in ["gemma", "deepseek"]):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": "You are a precise and efficient classification model."},
                {"role": "user", "content": prompt},
            ]

        # Generate using selected mode
        output = ""
        try:
            if llm_info["mode"] == "pipeline":
                output = generate_with_pipeline(llm_info["llm"], messages, max_retries=3, max_new_tokens=512)
            elif llm_info["mode"] == "model_tokenizer":
                output = generate_with_model_tokenizer(llm_info["model"], llm_info["tokenizer"], llm_info["device"], messages, max_new_tokens=512)
            else:
                raise RuntimeError("Unknown generation mode")
        except Exception as e:
            print("Generation failed:", e)
            output = ""

        # Try extraction
        try:
            response_dict = extract_first_dictionary(output)
            if response_dict:
                label = extract_labels(response_dict)
            else:
                label = extract_labels(output)
            print("->", label)
            results.append(label)
        except Exception as e:
            print(f"Decoding Error at sample {i}: {e}")
            print("Saving raw output.")
            results.append(output)

        sentence_list.append(sentence)
        # short delay to avoid throttling for remote services
        time.sleep(1)

    # Save results
    results_dir = os.path.join(root_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"{dataset_name}_{prompt_type}_shot.xlsx")

    if os.path.exists(output_file):
        data = pd.read_excel(output_file)
    else:
        data = pd.DataFrame({"sentence": sentence_list})

    data[llm_info["llm_name"]] = results
    data.to_excel(output_file, index=False)
    print(f"Saved results to: {output_file}")

# ----------------
# Main
# ----------------
def main(args):
    # Load datasets
    if args.dataset_name == "emo":
        dataset = pd.read_excel(os.path.join(dataset_dir, "emo_test.xlsx"))
    elif args.dataset_name == "senti":
        dataset = pd.read_excel(os.path.join(dataset_dir, "senti_test.xlsx"))
    elif args.dataset_name == "hate":
        dataset = pd.read_excel(os.path.join(dataset_dir, "hate_test.xlsx"))
    elif args.dataset_name == "fake":
        dataset = pd.read_excel(os.path.join(dataset_dir, "fake_test.xlsx"))
    else:
        raise ValueError("Unknown dataset_name")

    # Free memory
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    # Detect available backends in order: CUDA -> DirectML -> CPU
    use_cuda = torch.cuda.is_available()
    use_directml = False
    dml_device = None
    try:
        import importlib
        if importlib.util.find_spec("torch_directml") is not None:
            import torch_directml
            # create a DML device object
            try:
                dml_device = torch_directml.device()
                use_directml = True
            except Exception as e:
                print("torch_directml found but failed to create device:", e)
                use_directml = False
    except Exception:
        pass

    # warn about large models
    big_model_keywords = ["Mixtral", "mixtral", "Mistral-7B", "mistral-7b", "Mixtral-8x22B", "mixtral-8x22b", "7b", "8x22b"]
    if any(k.lower() in args.llm_id.lower() for k in big_model_keywords):
        print("WARNING: You asked for a large model. On your laptop this may OOM or be extremely slow. Consider using a smaller model locally or use Colab / remote GPU.")

    # Decide how to load and run the model
    llm_info = {"llm_name": args.llm_name}

    # -------------------------
    # CUDA path (NVIDIA GPU)
    # -------------------------
    if use_cuda:
        print("CUDA available — using transformers pipeline with GPU (device=0).")
        try:
            from transformers import pipeline
            pipe = pipeline("text-generation", model=args.llm_id, device=0)
            llm_info.update({"mode": "pipeline", "llm": pipe})
        except Exception as e:
            print("Failed to initialize pipeline on CUDA:", e)
            print("Falling back to CPU pipeline.")
            from transformers import pipeline
            pipe = pipeline("text-generation", model=args.llm_id, device=-1)
            llm_info.update({"mode": "pipeline", "llm": pipe})

    # -------------------------
    # DirectML path (AMD on Windows)
    # -------------------------
    elif use_directml and dml_device is not None:
        print("DirectML (torch_directml) detected — attempting to load model onto DirectML device.")
        try:
            # load tokenizer + model and move to dml device
            from transformers import AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
            # Try to use smaller memory footprint if possible
            model = AutoModelForCausalLM.from_pretrained(args.llm_id)
            # Move model to DirectML device (torch_directml.device())
            try:
                model.to(dml_device)
                device_for_generate = dml_device
                print("Model moved to DirectML device.")
            except Exception as e:
                print("Failed to move model to DirectML device (will try CPU):", e)
                device_for_generate = torch.device("cpu")

            llm_info.update({
                "mode": "model_tokenizer",
                "model": model,
                "tokenizer": tokenizer,
                "device": device_for_generate
            })
        except Exception as e:
            print("Failed to prepare model/tokenizer for DirectML:", e)
            print("Falling back to CPU pipeline.")
            from transformers import pipeline
            pipe = pipeline("text-generation", model=args.llm_id, device=-1)
            llm_info.update({"mode": "pipeline", "llm": pipe})

    # -------------------------
    # CPU fallback path
    # -------------------------
    else:
        print("No CUDA or DirectML detected — using CPU pipeline (may be slow).")
        from transformers import pipeline
        pipe = pipeline("text-generation", model=args.llm_id, device=-1)
        llm_info.update({"mode": "pipeline", "llm": pipe})

    # Run generation
    response_generation(dataset, args.dataset_name, args.prompt_type, llm_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero/Few Shot Classification (local with DirectML/CUDA detection)")
    parser.add_argument("--llm_id", dest="llm_id", type=str, required=True, help="Model id (HF path)")
    parser.add_argument("--llm_name", dest="llm_name", type=str, required=True, help="LLM name (short)")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str, required=True, help="Dataset name (emo/senti/hate/fake)")
    parser.add_argument("--prompt_type", dest="prompt_type", type=str, required=True, help="Prompt type (zero/few)")
    args = parser.parse_args()
    main(args)
