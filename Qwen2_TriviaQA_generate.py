"""
TriviaQA Answer Generation with Qwen2-1.5B

Loads TriviaQA data, generates answers for all samples, then splits into
three non-overlapping sets:
  - Set A (~500): training — train 336 per-head binary classifiers
  - Set B (~300): validation — evaluate classifier AUC, select top-k heads
  - Set C (~200): test — end-to-end evaluation (voting hallucination detection)

For each QA pair, generates an answer using greedy decoding, then auto-labels
as correct (if generated answer matches any ideal alias) or hallucination.

Output: CSV files with question, generated answer, ideal answers, and label.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm


def load_config(config_path="config.json"):
    with open(config_path) as f:
        return json.load(f)


def init_model(model_path: str, dtype=torch.bfloat16, device_map="auto"):
    """Load Qwen2-1.5B model and tokenizer from local path."""
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Override generation config to suppress sampling-related warnings
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False

    print(f"Model loaded. Hidden size={model.config.hidden_size}, "
          f"num_layers={model.config.num_hidden_layers}, "
          f"num_heads={model.config.num_attention_heads}")
    return model, tokenizer


def load_triviaqa_data(data_path: str):
    """Load TriviaQA data from local jsonl file."""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            samples.append(item)
    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def split_datasets(results, n_a=500, n_b=300, n_c=200, seed=42):
    """
    Split labeled results into three non-overlapping sets.
    Set A: training set for 336 per-head binary classifiers.
    Set B: validation set for evaluating classifier AUC and selecting top-k heads.
    Set C: test set for end-to-end voting hallucination detection evaluation.

    Args:
        results: list of dicts with 'label' key (1=correct, 0=hallucination)
        n_a: samples for set A (training)
        n_b: samples for set B (validation)
        n_c: samples for set C (test)
        seed: random seed

    Returns:
        set_a, set_b, set_c: lists of result dicts
    """
    rng = np.random.RandomState(seed)

    # Shuffle all results (no need to balance — classifiers handle imbalance)
    all_results = list(results)
    rng.shuffle(all_results)

    set_a = all_results[:n_a]
    set_b = all_results[n_a:n_a + n_b]
    set_c = all_results[n_a + n_b:n_a + n_b + n_c]

    correct_a = sum(1 for r in set_a if r['label'] == 1)
    correct_b = sum(1 for r in set_b if r['label'] == 1)
    correct_c = sum(1 for r in set_c if r['label'] == 1)

    print(f"Split: Set A={len(set_a)} ({correct_a} correct), "
          f"Set B={len(set_b)} ({correct_b} correct), "
          f"Set C={len(set_c)} ({correct_c} correct)")
    return set_a, set_b, set_c


def is_correct(generated_answer: str, ideal_answers: list) -> bool:
    """
    Check if generated answer matches any ideal answer (fuzzy matching).
    Uses substring matching after lowercasing and stripping.
    """
    gen = generated_answer.strip().lower()
    if not gen:
        return False
    for gt in ideal_answers:
        gt_lower = gt.strip().lower()
        if gt_lower in gen or gen in gt_lower:
            return True
    return False


def generate_answer(model, tokenizer, messages: list, max_new_tokens: int = 50):
    """
    Generate answer for a QA pair using greedy decoding.

    Args:
        model: The loaded causal LM.
        tokenizer: The tokenizer.
        messages: List of chat messages [{"role": ..., "content": ...}, ...].
        max_new_tokens: Maximum tokens to generate.

    Returns:
        generated_text: The generated answer string.
        input_length: Number of input tokens (for later position tracking).
    """
    device = next(model.parameters()).device

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            temperature=None,  # suppress warning: unused in greedy mode
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (exclude input)
    generated_ids = output_ids[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip(), input_length


def generate_and_label(model, tokenizer, samples: list,
                       max_new_tokens: int = 50, batch_report: int = 50):
    """
    Generate answers for all samples and auto-label them.

    Returns:
        results: list of dicts with question, generated_answer, ideal, label, etc.
    """
    results = []
    correct_count = 0
    hallucination_count = 0

    for idx, sample in enumerate(tqdm(samples, desc="Generating answers")):
        messages = sample["input"]
        ideal = sample["ideal"]

        # Generate answer
        generated_answer, input_length = generate_answer(
            model, tokenizer, messages, max_new_tokens=max_new_tokens
        )

        # Auto-label
        label = is_correct(generated_answer, ideal)
        if label:
            correct_count += 1
        else:
            hallucination_count += 1

        # Extract question text for readability
        question = ""
        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"]
                break

        results.append({
            "question": question,
            "generated_answer": generated_answer,
            "ideal": json.dumps(ideal),
            "label": int(label),  # 1=correct, 0=hallucination
            "input_length": input_length,
        })

        if (idx + 1) % batch_report == 0:
            total = correct_count + hallucination_count
            print(f"  [{idx+1}/{len(samples)}] "
                  f"Correct: {correct_count} ({correct_count/total:.1%}), "
                  f"Hallucination: {hallucination_count} ({hallucination_count/total:.1%})")

    total = len(results)
    print(f"\nGeneration complete: {total} samples")
    print(f"  Correct: {correct_count} ({correct_count/total:.1%})")
    print(f"  Hallucination: {hallucination_count} ({hallucination_count/total:.1%})")

    return results


def run_generation(config_path="config.json",
                   data_path=None,
                   output_dir=None,
                   n_a=500, n_b=300, n_c=200,
                   max_samples=2000,
                   max_new_tokens=50):
    """Run the full TriviaQA generation and labeling pipeline."""
    config = load_config(config_path)

    model_path = config.get("model_path", "model")
    dtype_str = config.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device_map = config.get("device_map", "auto")

    if data_path is None:
        data_path = "datasets/triviaqa/trivia_qa/test.jsonl"
    if output_dir is None:
        output_dir = config.get("processed_dataset_path", "processed_datasets")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = init_model(model_path, dtype=dtype, device_map=device_map)

    # Load data and subsample if needed
    all_samples = load_triviaqa_data(data_path)
    if max_samples and max_samples < len(all_samples):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_samples), size=max_samples, replace=False)
        all_samples = [all_samples[i] for i in indices]
        print(f"Subsampled to {max_samples} samples for generation")

    # Generate answers for ALL samples first, then split into three sets
    print(f"\n{'='*60}")
    print(f"Generating answers for all {len(all_samples)} samples")
    print(f"{'='*60}")

    all_results = generate_and_label(model, tokenizer, all_samples,
                                     max_new_tokens=max_new_tokens)

    # Split labeled results into train / validation / test sets
    set_a, set_b, set_c = split_datasets(all_results, n_a=n_a, n_b=n_b, n_c=n_c)

    # Save each set to CSV
    for set_name, set_data in [("set_a", set_a), ("set_b", set_b), ("set_c", set_c)]:
        df = pd.DataFrame(set_data)
        output_file = output_dir / f"triviaqa_{set_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {set_name}: {len(df)} samples to {output_file}")
        correct = df['label'].sum()
        total = len(df)
        print(f"  Distribution: {correct} correct, {total - correct} hallucination")

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()
    print("\nDone! All sets generated and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TriviaQA answer generation with Qwen2-1.5B"
    )
    parser.add_argument("--config", default="config.json",
                        help="Path to config file")
    parser.add_argument("--data_path", default=None,
                        help="Path to TriviaQA jsonl file")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for CSV files")
    parser.add_argument("--n_a", type=int, default=500,
                        help="Number of samples for set A (training)")
    parser.add_argument("--n_b", type=int, default=300,
                        help="Number of samples for set B (validation)")
    parser.add_argument("--n_c", type=int, default=200,
                        help="Number of samples for set C (evaluation)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Max tokens to generate per answer")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max samples to load from dataset (default 2000, set 0 for all)")
    args = parser.parse_args()

    run_generation(
        config_path=args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_a=args.n_a,
        n_b=args.n_b,
        n_c=args.n_c,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        max_new_tokens=args.max_new_tokens,
    )
